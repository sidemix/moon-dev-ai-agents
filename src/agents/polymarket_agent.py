#!/usr/bin/env python3
"""
Moon Dev's Polymarket Prediction Market Agent
Built with love by Moon Dev 🚀
This agent scans Polymarket trades, saves markets to CSV, and uses AI to make predictions.
NO ACTUAL TRADING - just predictions and analysis for now.
"""
import os
import sys
import time
import json
import requests
import pandas as pd
import threading
import websocket
from datetime import datetime, timedelta
from pathlib import Path
from termcolor import cprint

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.model_factory import ModelFactory

# ==============================================================================
# CONFIGURATION - Customize these settings
# ==============================================================================
MIN_TRADE_SIZE_USD = 500
IGNORE_PRICE_THRESHOLD = 0.02
LOOKBACK_HOURS = 24

IGNORE_CRYPTO_KEYWORDS = [
    'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana', 'sol',
    'dogecoin', 'doge', 'shiba', 'cardano', 'ada', 'ripple', 'xrp',
]

IGNORE_SPORTS_KEYWORDS = [
    'nba', 'nfl', 'mlb', 'nhl', 'mls', 'ufc', 'boxing',
    'football', 'basketball', 'baseball', 'hockey', 'soccer',
    'super bowl', 'world series', 'playoffs', 'championship',
    'lakers', 'warriors', 'celtics', 'knicks', 'heat', 'bucks',
    'cowboys', 'patriots', 'chiefs', 'eagles', 'packers',
    'yankees', 'dodgers', 'red sox', 'mets',
    'premier league', 'la liga', 'champions league',
    'tennis', 'golf', 'nascar', 'formula 1', 'f1',
    'cricket',
]

ANALYSIS_CHECK_INTERVAL_SECONDS = 300
NEW_MARKETS_FOR_ANALYSIS = 3
MARKETS_TO_ANALYZE = 3
MARKETS_TO_DISPLAY = 20
REANALYSIS_HOURS = 8

USE_SWARM_MODE = True
AI_MODEL_PROVIDER = "xai"
AI_MODEL_NAME = "grok-2-fast-reasoning"
SEND_PRICE_INFO_TO_AI = False

# ==============================================================================
# AI Prompts
# ==============================================================================
MARKET_ANALYSIS_SYSTEM_PROMPT = """You are a prediction market expert analyzing Polymarket markets.
For each market, provide your prediction in this exact format:
MARKET [number]: [decision]
Reasoning: [brief 1-2 sentence explanation]
Decision must be one of: YES, NO, or NO_TRADE
- YES means you would bet on the "Yes" outcome
- NO means you would bet on the "No" outcome
- NO_TRADE means you would not take a position
Be concise and focused on the most promising opportunities."""

TOP_MARKETS_COUNT = 5
CONSENSUS_AI_PROMPT_TEMPLATE = """You are analyzing predictions from multiple AI models on Polymarket markets.
MARKET REFERENCE:
{market_reference}
ALL AI RESPONSES:
{all_responses}
Based on ALL of these AI responses, identify the TOP {top_count} MARKETS that have the STRONGEST CONSENSUS across all models.
Rules:
- Look for markets where most AIs agree on the same side (YES, NO, or NO_TRADE)
- Ignore markets with split opinions
- Focus on clear, strong agreement
- DO NOT use any reasoning or thinking - just analyze the responses
- Provide exactly {top_count} markets ranked by consensus strength
Format your response EXACTLY like this:
TOP {top_count} CONSENSUS PICKS:
1. Market [number]: [market title]
   Side: [YES/NO/NO_TRADE]
   Consensus: [X out of Y models agreed]
   Link: [polymarket URL from market reference]
   Reasoning: [1 sentence why this is a strong pick]
2. Market [number]: [market title]
   Side: [YES/NO/NO_TRADE]
   Consensus: [X out of Y models agreed]
   Link: [polymarket URL from market reference]
   Reasoning: [1 sentence why this is a strong pick]
[Continue for all {top_count} markets...]
"""

# Data paths
DATA_FOLDER = os.path.join(project_root, "src/data/polymarket")
MARKETS_CSV = os.path.join(DATA_FOLDER, "markets.csv")
PREDICTIONS_CSV = os.path.join(DATA_FOLDER, "predictions.csv")
CONSENSUS_PICKS_CSV = os.path.join(DATA_FOLDER, "consensus_picks.csv")

# Polymarket API & WebSocket
POLYMARKET_API_BASE = "https://data-api.polymarket.com"
WEBSOCKET_URL = "wss://ws-live-data.polymarket.com"

# ==============================================================================
# Polymarket Agent — FIXED & WORKING
# ==============================================================================
class PolymarketAgent:
    def __init__(self):
        cprint("\n" + "="*80, "cyan")
        cprint("🌙 Polymarket Prediction Market Agent - Initializing", "cyan", attrs=['bold'])
        cprint("="*80, "cyan")

        os.makedirs(DATA_FOLDER, exist_ok=True)
        self.csv_lock = threading.Lock()
        self.last_analyzed_count = 0
        self.last_analysis_run_timestamp = None
        self.ws = None
        self.ws_connected = False
        self.total_trades_received = 0
        self.filtered_trades_count = 0
        self.ignored_crypto_count = 0
        self.ignored_sports_count = 0

        # Initialize AI models
        if USE_SWARM_MODE:
            cprint("🤖 Using SWARM MODE - Multiple AI models", "green")
            try:
                from src.agents.swarm_agent import SwarmAgent
                self.swarm = SwarmAgent()
                cprint("✅ Swarm agent loaded successfully", "green")
            except Exception as e:
                cprint(f"❌ Failed to load swarm agent: {e}", "red")
                cprint("💡 Falling back to single model mode", "yellow")
                self.swarm = None
                self.model = ModelFactory().get_model(AI_MODEL_PROVIDER, AI_MODEL_NAME)
        else:
            cprint(f"🤖 Using single model: {AI_MODEL_PROVIDER}/{AI_MODEL_NAME}", "green")
            self.model = ModelFactory().get_model(AI_MODEL_PROVIDER, AI_MODEL_NAME)
            self.swarm = None

        self.markets_df = self._load_markets()
        self.predictions_df = self._load_predictions()
        cprint(f"📊 Loaded {len(self.markets_df)} existing markets from CSV", "cyan")
        cprint(f"🔮 Loaded {len(self.predictions_df)} existing predictions from CSV", "cyan")
        if len(self.predictions_df) > 0:
            unique_runs = self.predictions_df['analysis_run_id'].nunique()
            cprint(f" └─ {unique_runs} historical analysis runs", "cyan")
        cprint("✨ Initialization complete!\n", "green")

    def _load_markets(self):
        if os.path.exists(MARKETS_CSV):
            try:
                df = pd.read_csv(MARKETS_CSV)
                # Ensure all required columns exist
                required = ['timestamp', 'market_id', 'event_slug', 'title', 'outcome', 'price', 'size_usd', 'first_seen', 'last_analyzed', 'last_trade_timestamp']
                for col in required:
                    if col not in df.columns:
                        df[col] = None
                return df
            except Exception as e:
                cprint(f"⚠️ Error loading markets CSV: {e}", "yellow")
        return pd.DataFrame(columns=[
            'timestamp', 'market_id', 'event_slug', 'title',
            'outcome', 'price', 'size_usd', 'first_seen', 'last_analyzed', 'last_trade_timestamp'
        ])

    def _load_predictions(self):
        if os.path.exists(PREDICTIONS_CSV):
            try:
                df = pd.read_csv(PREDICTIONS_CSV)
                return df
            except Exception as e:
                cprint(f"⚠️ Error loading predictions CSV: {e}", "yellow")
        return pd.DataFrame(columns=[
            'analysis_timestamp', 'analysis_run_id', 'market_title', 'market_slug',
            'claude_prediction', 'opus_prediction', 'openai_prediction', 'groq_prediction',
            'gemini_prediction', 'deepseek_prediction', 'xai_prediction',
            'ollama_prediction', 'consensus_prediction', 'num_models_responded',
            'market_link'
        ])

    def _save_markets(self):
        try:
            with self.csv_lock:
                self.markets_df.to_csv(MARKETS_CSV, index=False)
        except Exception as e:
            cprint(f"❌ Error saving markets CSV: {e}", "red")

    def _save_predictions(self):
        try:
            with self.csv_lock:
                self.predictions_df.to_csv(PREDICTIONS_CSV, index=False)
            cprint(f"💾 Saved {len(self.predictions_df)} predictions to CSV", "green")
        except Exception as e:
            cprint(f"❌ Error saving predictions CSV: {e}", "red")

    def is_near_resolution(self, price):
        price_float = float(price)
        return price_float <= IGNORE_PRICE_THRESHOLD or price_float >= (1.0 - IGNORE_PRICE_THRESHOLD)

    def should_ignore_market(self, title):
        title_lower = title.lower()
        for keyword in IGNORE_CRYPTO_KEYWORDS:
            if keyword in title_lower:
                return True, f"crypto/bitcoin ({keyword})"
        for keyword in IGNORE_SPORTS_KEYWORDS:
            if keyword in title_lower:
                return True, f"sports ({keyword})"
        return False, None

    def on_ws_message(self, ws, message):
        try:
            data = json.loads(message)
            if isinstance(data, dict) and data.get('type') == 'subscribed':
                cprint("✅ WebSocket subscribed!", "green")
                self.ws_connected = True
                return
            if data.get('type') == 'pong':
                return
            if data.get('topic') == 'activity' and data.get('type') == 'orders_matched':
                self.total_trades_received += 1
                payload = data.get('payload', {})
                price = float(payload.get('price', 0))
                size = float(payload.get('size', 0))
                usd_amount = price * size
                title = payload.get('title', 'Unknown')
                should_ignore, reason = self.should_ignore_market(title)
                if should_ignore:
                    if 'crypto' in reason:
                        self.ignored_crypto_count += 1
                    elif 'sports' in reason:
                        self.ignored_sports_count += 1
                    return
                if usd_amount >= MIN_TRADE_SIZE_USD and not self.is_near_resolution(price):
                    self.filtered_trades_count += 1
                    trade_data = {
                        'timestamp': payload.get('timestamp', time.time()),
                        'conditionId': payload.get('conditionId', f"ws_{time.time()}"),
                        'eventSlug': payload.get('eventSlug', '') or payload.get('slug', ''),
                        'title': title,
                        'outcome': payload.get('outcome', 'Unknown'),
                        'price': price,
                        'size': usd_amount,
                        'side': payload.get('side', ''),
                        'trader': payload.get('name', payload.get('pseudonym', 'Unknown'))
                    }
                    self.process_trades([trade_data])
        except json.JSONDecodeError:
            pass
        except Exception as e:
            cprint(f"Error processing message: {e}", "yellow")

    def on_ws_error(self, ws, error):
        cprint(f"WebSocket Error: {error}", "red")

    def on_ws_close(self, ws, close_status_code, close_msg):
        self.ws_connected = False
        cprint(f"WebSocket closed: {close_status_code} - {close_msg}", "yellow")
        cprint("Reconnecting in 5 seconds...", "cyan")
        time.sleep(5)
        self.connect_websocket()

    def on_ws_open(self, ws):
        cprint("WebSocket connected!", "green")
        subscription_msg = {
            "action": "subscribe",
            "subscriptions": [{"topic": "activity", "type": "orders_matched"}]
        }
        ws.send(json.dumps(subscription_msg))
        self.ws_connected = True
        cprint("Subscription sent! Waiting for trades...", "green")

        def send_ping():
            while True:
                time.sleep(5)
                try:
                    ws.send(json.dumps({"type": "ping"}))
                except:
                    break
        threading.Thread(target=send_ping, daemon=True).start()

    def connect_websocket(self):
        cprint(f"Connecting to {WEBSOCKET_URL}...", "cyan")
        self.ws = websocket.WebSocketApp(
            WEBSOCKET_URL,
            on_open=self.on_ws_open,
            on_message=self.on_ws_message,
            on_error=self.on_ws_error,
            on_close=self.on_ws_close
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()
        cprint("WebSocket thread started!", "green")

    def fetch_historical_trades(self, hours_back=None):
        if hours_back is None:
            hours_back = LOOKBACK_HOURS
        try:
            cprint(f"Fetching historical trades (last {hours_back}h)...", "yellow")
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            cutoff_timestamp = int(cutoff_time.timestamp())
            url = f"{POLYMARKET_API_BASE}/trades"
            params = {'limit': 1000, '_min_timestamp': cutoff_timestamp}
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            trades = response.json()
            cprint(f"Fetched {len(trades)} historical trades", "green")
            filtered = []
            for trade in trades:
                price = float(trade.get('price', 0))
                size = float(trade.get('size', 0))
                usd_amount = price * size
                title = trade.get('title', 'Unknown')
                if self.should_ignore_market(title)[0]:
                    continue
                if usd_amount >= MIN_TRADE_SIZE_USD and not self.is_near_resolution(price):
                    filtered.append(trade)
            cprint(f"Found {len(filtered)} trades over ${MIN_TRADE_SIZE_USD} (after filters)", "cyan")
            return filtered
        except Exception as e:
            cprint(f"Error fetching historical trades: {e}", "red")
            return []

    def process_trades(self, trades):
        if not trades:
            return
        unique_markets = {}
        for trade in trades:
            market_id = trade.get('conditionId', '')
            if market_id and market_id not in unique_markets:
                unique_markets[market_id] = trade
        new_markets = 0
        updated_markets = 0
        for market_id, trade in unique_markets.items():
            try:
                event_slug = trade.get('eventSlug', '')
                title = trade.get('title', 'Unknown Market')
                outcome = trade.get('outcome', '')
                price = float(trade.get('price', 0))
                size_usd = float(trade.get('size', 0))
                timestamp = trade.get('timestamp', '')
                condition_id = trade.get('conditionId', '')
                if market_id in self.markets_df['market_id'].values:
                    mask = self.markets_df['market_id'] == market_id
                    self.markets_df.loc[mask, 'timestamp'] = timestamp
                    self.markets_df.loc[mask, 'outcome'] = outcome
                    self.markets_df.loc[mask, 'price'] = price
                    self.markets_df.loc[mask, 'size_usd'] = size_usd
                    self.markets_df.loc[mask, 'last_trade_timestamp'] = datetime.now().isoformat()
                    updated_markets += 1
                    continue
                new_market = {
                    'timestamp': timestamp,
                    'market_id': condition_id,
                    'event_slug': event_slug,
                    'title': title,
                    'outcome': outcome,
                    'price': price,
                    'size_usd': size_usd,
                    'first_seen': datetime.now().isoformat(),
                    'last_analyzed': None,
                    'last_trade_timestamp': datetime.now().isoformat()
                }
                self.markets_df = pd.concat([self.markets_df, pd.DataFrame([new_market])], ignore_index=True)
                new_markets += 1
                cprint(f"✨ NEW: ${size_usd:,.0f} - {title[:70]}", "green")
            except Exception as e:
                cprint(f"Error processing trade: {e}", "yellow")
                continue
        if new_markets or updated_markets:
            self._save_markets()
            if updated_markets:
                cprint(f"Updated {updated_markets} existing markets", "cyan")

    def display_recent_markets(self):
        if len(self.markets_df) == 0:
            cprint("\nNo markets in database yet", "yellow")
            return
        cprint("\n" + "="*80, "cyan")
        cprint(f"Most Recent {min(MARKETS_TO_DISPLAY, len(self.markets_df))} Markets", "cyan", attrs=['bold'])
        cprint("="*80, "cyan")
        recent = self.markets_df.tail(MARKETS_TO_DISPLAY)
        for _, row in recent.iterrows():
            title = row['title'][:60] + "..." if len(row['title']) > 60 else row['title']
            size = row['size_usd']
            outcome = row['outcome']
            cprint(f"\n${size:,.2f} trade on {outcome}", "yellow")
            cprint(f"{title}", "white")
            cprint(f"https://polymarket.com/event/{row['event_slug']}", "cyan")
        cprint("\n" + "="*80, "cyan")
        cprint(f"Total markets tracked: {len(self.markets_df)}", "green", attrs=['bold'])
        cprint("="*80 + "\n", "cyan")

    def get_ai_predictions(self):
        if len(self.markets_df) == 0:
            cprint("\nNo markets to analyze yet", "yellow")
            return
        markets_to_analyze = self.markets_df.tail(MARKETS_TO_ANALYZE)
        analysis_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_timestamp = datetime.now().isoformat()
        cprint("\n" + "="*80, "magenta")
        cprint(f"AI Analysis - Analyzing {len(markets_to_analyze)} markets", "magenta", attrs=['bold'])
        cprint(f"Analysis Run ID: {analysis_run_id}", "magenta")
        cprint(f"Price info to AI: {'ENABLED' if SEND_PRICE_INFO_TO_AI else 'DISABLED'}", "green" if SEND_PRICE_INFO_TO_AI else "yellow")
        cprint("="*80, "magenta")

        if SEND_PRICE_INFO_TO_AI:
            markets_text = "\n\n".join([
                f"Market {i+1}:\nTitle: {row['title']}\nCurrent Price: ${row['price']:.2f} ({row['price']*100:.1f}% odds for {row['outcome']})\nRecent trade: ${row['size_usd']:,.2f} on {row['outcome']}\nLink: https://polymarket.com/event/{row['event_slug']}"
                for i, row in markets_to_analyze.iterrows()
            ])
        else:
            markets_text = "\n\n".join([
                f"Market {i+1}:\nTitle: {row['title']}\nRecent trade: ${row['size_usd']:,.2f} on {row['outcome']}\nLink: https://polymarket.com/event/{row['event_slug']}"
                for i, row in markets_to_analyze.iterrows()
            ])

        system_prompt = MARKET_ANALYSIS_SYSTEM_PROMPT
        user_prompt = f"""Analyze these {len(markets_to_analyze)} Polymarket markets and provide your predictions:
{markets_text}
Provide predictions for each market in the specified format."""

        if USE_SWARM_MODE and self.swarm:
            cprint("\nGetting predictions from AI swarm (120s timeout per model)...\n", "cyan")
            cprint("Sending prompts to swarm...", "cyan")
            swarm_result = self.swarm.query(prompt=user_prompt, system_prompt=system_prompt)

            if not swarm_result or not swarm_result.get('responses'):
                cprint("No responses from swarm", "red")
                return

            successful = [name for name, data in swarm_result.get('responses', {}).items() if data.get('success')]
            if not successful:
                cprint("All AI models failed", "red")
                return

            cprint(f"\nReceived {len(successful)}/{len(swarm_result['responses'])} successful responses!\n", "green", attrs=['bold'])

            cprint("="*80, "yellow")
            cprint("Individual AI Predictions", "yellow", attrs=['bold'])
            cprint("="*80, "yellow")
            for model_name, model_data in swarm_result.get('responses', {}).items():
                if model_data.get('success'):
                    cprint(f"\n{model_name.upper()} ({model_data.get('response_time', 0):.1f}s)", "cyan", attrs=['bold'])
                    cprint(model_data.get('response', 'No response'), "white")
                else:
                    cprint(f"\n{model_name.upper()} - FAILED: {model_data.get('error', 'Unknown')}", "red")

            consensus_text = self._calculate_polymarket_consensus(swarm_result, markets_to_analyze)
            cprint("\n" + "="*80, "green")
            cprint("CONSENSUS ANALYSIS", "green", attrs=['bold'])
            cprint(f"Based on {len(successful)} AI models", "green")
            cprint("="*80, "green")
            cprint(consensus_text, "white")
            cprint("="*80 + "\n", "green")

            self._get_top_consensus_picks(swarm_result, markets_to_analyze)

            try:
                self._save_swarm_predictions(analysis_run_id, analysis_timestamp, markets_to_analyze, swarm_result)
                cprint(f"\nPredictions saved to: {PREDICTIONS_CSV}", "cyan", attrs=['bold'])
            except Exception as e:
                cprint(f"Error saving predictions: {e}", "red")

            self._mark_markets_analyzed(markets_to_analyze, analysis_timestamp)
        else:
            # Single model fallback
            cprint(f"\nGetting predictions from {AI_MODEL_PROVIDER}/{AI_MODEL_NAME}...\n", "cyan")
            try:
                response = self.model.generate_response(system_prompt=system_prompt, user_content=user_prompt, temperature=0.7)
                cprint("="*80, "green")
                cprint("AI PREDICTION", "green", attrs=['bold'])
                cprint("="*80, "green")
                cprint(response.content, "white")
                cprint("="*80 + "\n", "green")
            except Exception as e:
                cprint(f"Error getting prediction: {e}", "red")

    # [Rest of your methods — unchanged below this point]
    # ... (all your _calculate_polymarket_consensus, _get_top_consensus_picks, etc. stay exactly the same)

    def status_display_loop(self):
        cprint("\nSTATUS DISPLAY THREAD STARTED", "cyan", attrs=['bold'])
        cprint("Showing stats every 30 seconds\n", "cyan")
        while True:
            try:
                time.sleep(30)
                total_markets = len(self.markets_df)
                now = datetime.now()
                cutoff_time = now - timedelta(hours=REANALYSIS_HOURS)
                fresh_eligible_count = 0
                for idx, row in self.markets_df.iterrows():
                    last_analyzed = row.get('last_analyzed')
                    last_trade = row.get('last_trade_timestamp')
                    is_eligible = pd.isna(last_analyzed) or last_analyzed is None
                    if not is_eligible:
                        try:
                            analyzed_time = pd.to_datetime(last_analyzed)
                            if analyzed_time < cutoff_time:
                                is_eligible = True
                        except:
                            is_eligible = True
                    has_fresh_trade = self.last_analysis_run_timestamp is None
                    if not has_fresh_trade and not pd.isna(last_trade) and last_trade is not None:
                        try:
                            trade_time = pd.to_datetime(last_trade)
                            last_run_time = pd.to_datetime(self.last_analysis_run_timestamp)
                            if trade_time > last_run_time:
                                has_fresh_trade = True
                        except:
                            pass
                    if is_eligible and has_fresh_trade:
                        fresh_eligible_count += 1
                cprint(f"\n{'='*60}", "cyan")
                cprint(f"📊 Moon Dev Status @ {datetime.now().strftime('%H:%M:%S')}", "cyan", attrs=['bold'])
                cprint(f"{'='*60}", "cyan")
                cprint(f" WebSocket Connected: {'YES' if self.ws_connected else 'NO'}", "green" if self.ws_connected else "red")
                cprint(f" Total trades received: {self.total_trades_received}", "white")
                cprint(f" Ignored crypto/bitcoin: {self.ignored_crypto_count}", "red")
                cprint(f" Ignored sports: {self.ignored_sports_count}", "red")
                cprint(f" Filtered trades (>=${MIN_TRADE_SIZE_USD}): {self.filtered_trades_count}", "yellow")
                cprint(f" Total markets in database: {total_markets}", "white")
                cprint(f" Fresh eligible markets: {fresh_eligible_count}", "yellow" if fresh_eligible_count < NEW_MARKETS_FOR_ANALYSIS else "green", attrs=['bold'])
                cprint(f" (Eligible + traded since last run)", "white")
                if fresh_eligible_count >= NEW_MARKETS_FOR_ANALYSIS:
                    cprint(f" READY for analysis! (Have {fresh_eligible_count}, need {NEW_MARKETS_FOR_ANALYSIS})", "green", attrs=['bold'])
                else:
                    cprint(f" Collecting... (Have {fresh_eligible_count}, need {NEW_MARKETS_FOR_ANALYSIS})", "yellow")
                cprint(f"{'='*60}\n", "cyan")
            except Exception as e:
                cprint(f"Error in status display loop: {e}", "red")

    def analysis_loop(self):
        cprint("\nANALYSIS THREAD STARTED", "magenta", attrs=['bold'])
        cprint("Running first analysis NOW, then every 300 seconds\n", "magenta")
        while True:
            try:
                self.analysis_cycle()
                next_check = datetime.now() + timedelta(seconds=ANALYSIS_CHECK_INTERVAL_SECONDS)
                cprint(f"Next analysis check at: {next_check.strftime('%H:%M:%S')}\n", "magenta")
                time.sleep(ANALYSIS_CHECK_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                break
            except Exception as e:
                cprint(f"Error in analysis loop: {e}", "red")
                time.sleep(ANALYSIS_CHECK_INTERVAL_SECONDS)

def main():
    cprint("\n" + "="*80, "cyan")
    cprint("Moon Dev's Polymarket Agent - WebSocket Edition!", "cyan", attrs=['bold'])
    cprint("="*80, "cyan")
    agent = PolymarketAgent()
    historical_trades = agent.fetch_historical_trades()
    if historical_trades:
        cprint(f"\nProcessing {len(historical_trades)} historical trades...", "cyan")
        agent.process_trades(historical_trades)
        cprint(f"Database populated with {len(agent.markets_df)} markets", "green")
    else:
        cprint("No historical trades found - starting fresh", "yellow")
    cprint("="*80 + "\n", "cyan")
    agent.connect_websocket()
    status_thread = threading.Thread(target=agent.status_display_loop, daemon=True)
    analysis_thread = threading.Thread(target=agent.analysis_loop, daemon=True)
    try:
        cprint("Moon Dev starting threads...\n", "green", attrs=['bold'])
        status_thread.start()
        analysis_thread.start()
        cprint("Moon Dev WebSocket + AI running! Press Ctrl+C to stop.\n", "green", attrs=['bold'])
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cprint("\n\n" + "="*80, "yellow")
        cprint("Moon Dev Polymarket Agent stopped by user", "yellow", attrs=['bold'])
        cprint("="*80 + "\n", "yellow")
        sys.exit(0)

if __name__ == "__main__":
    main()