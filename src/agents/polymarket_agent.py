#!/usr/bin/env python3
"""
Moon Dev's Polymarket Agent — FINAL 6-MODEL + TELEGRAM VERSION (Dec 2025)
Full 6-model swarm with adjustable Telegram threshold via env var
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

from src.models.model_factory import model_factory

# ==============================================================================
# CONFIGURATION - YOU CONTROL EVERYTHING FROM RAILWAY VARIABLES
# ==============================================================================

MIN_TRADE_SIZE_USD = int(os.getenv("MIN_TRADE_SIZE_USD", "500"))
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "24"))
ANALYSIS_CHECK_INTERVAL_SECONDS = int(os.getenv("ANALYSIS_INTERVAL", "300"))  # 5 min default
NEW_MARKETS_FOR_ANALYSIS = int(os.getenv("NEW_MARKETS_FOR_ANALYSIS", "3"))
MIN_AGREEMENT_FOR_ALERT = int(os.getenv("MIN_AGREEMENT_FOR_ALERT", "4"))  # ← CHANGE THIS!

# Telegram (optional)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Data paths
DATA_FOLDER = Path(project_root) / "src" / "data" / "polymarket"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
MARKETS_CSV = DATA_FOLDER / "markets.csv"
PREDICTIONS_CSV = DATA_FOLDER / "predictions.csv"

# ==============================================================================
# Polymarket Agent
# ==============================================================================

class PolymarketAgent:
    def __init__(self):
        cprint("\n" + "="*80, "cyan")
        cprint("Moon Dev's Polymarket Agent - 6-Model Swarm Edition", "cyan", attrs=['bold'])
        cprint("="*80, "cyan")

        self.csv_lock = threading.Lock()
        self.last_analysis_run_timestamp = None
        self.ws_connected = False

        # Load data
        self.markets_df = self._load_csv(MARKETS_CSV, ["market_id", "title", "event_slug", "last_trade"])
        self.predictions_df = self._load_csv(PREDICTIONS_CSV, ["analysis_run_id", "market_title", "consensus"])

        cprint(f"Loaded {len(self.markets_df)} markets | {len(self.predictions_df)} past predictions", "green")

        # WebSocket
        self.connect_websocket()

        # Start threads
        threading.Thread(target=self.status_display_loop, daemon=True).start()
        threading.Thread(target=self.analysis_loop, daemon=True).start()

        cprint("All threads running! Press Ctrl+C to stop.\n", "green", attrs=['bold'])

    def _load_csv(self, path, columns):
        if path.exists():
            try:
                return pd.read_csv(path)
            except:
                pass
        return pd.DataFrame(columns=columns)

    def _save_csv(self, df, path):
        with self.csv_lock:
            df.to_csv(path, index=False)

    def connect_websocket(self):
        cprint(f"Connecting to {websocket_url}...", "cyan")
        self.ws = websocket.WebSocketApp(
            "wss://ws-live-data.polymarket.com",
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=lambda ws, err: cprint(f"WS Error: {err}", "red"),
            on_close=lambda ws, *args: (time.sleep(5), self.connect_websocket())
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def on_open(self, ws):
        cprint("WebSocket Connected!", "green")
        self.ws_connected = True
        ws.send(json.dumps({"action": "subscribe", "subscriptions": [{"topic": "activity", "type": "orders_matched"}]}))

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get("type") != "orders_matched":
                return
            payload = data.get("payload", {})
            size = float(payload.get("size", 0))
            price = float(payload.get("price", 0))
            usd = size * price
            if usd < MIN_TRADE_SIZE_USD:
                return
            market_id = payload.get("conditionId")
            title = payload.get("title", "Unknown")
            slug = payload.get("eventSlug", "")
            if not market_id or not slug:
                return

            # Update or add market
            if market_id in self.markets_df["market_id"].values:
                self.markets_df.loc[self.markets_df["market_id"] == market_id, "last_trade"] = datetime.now().isoformat()
            else:
                new_row = pd.DataFrame([{
                    "market_id": market_id,
                    "title": title,
                    "event_slug": slug,
                    "last_trade": datetime.now().isoformat()
                }])
                self.markets_df = pd.concat([self.markets_df, new_row], ignore_index=True)
                cprint(f"NEW: ${usd:,.0f} → {title[:60]}", "green")

            self._save_csv(self.markets_df, MARKETS_CSV)

        except Exception as e:
            cprint(f"Error processing trade: {e}", "red")

    def status_display_loop(self):
        while True:
            time.sleep(30)
            fresh = len(self.markets_df[
                (pd.to_datetime(self.markets_df["last_trade"]) > (datetime.now() - timedelta(hours=8))) |
                self.markets_df["last_trade"].isna()
            ])
            cprint(f"\nStatus @ {datetime.now().strftime('%H:%M:%S')} | Markets: {len(self.markets_df)} | Fresh: {fresh}", "cyan")

    def analysis_loop(self):
        cprint("First analysis starting NOW...", "yellow", attrs=['bold'])
        while True:
            self.run_analysis()
            time.sleep(ANALYSIS_CHECK_INTERVAL_SECONDS)

    def run_analysis(self):
        # Count fresh eligible markets
        cutoff = datetime.now() - timedelta(hours=8) if self.last_analysis_run_timestamp else None
        mask = self.markets_df["last_trade"].isna() | (pd.to_datetime(self.markets_df["last_trade"]) > cutoff)
        fresh_count = len(self.markets_df[mask])

        if fresh_count < NEW_MARKETS_FOR_ANALYSIS and self.last_analysis_run_timestamp:
            return

        cprint(f"\nRUNNING 6-MODEL ANALYSIS on {fresh_count} fresh markets!", "magenta", attrs=['bold'])

        # Get top fresh markets
        markets = self.markets_df[mask].tail(10)
        prompt = "Analyze these Polymarket markets. For each, answer only: YES, NO, or HOLD\n\n" + "\n".join([
            f"{i+1}. {row['title']}\nLink: https://polymarket.com/event/{row['event_slug']}"
            for i, row in enumerate(markets.iterrows(), 1)
        ])

        try:
            result = model_factory.swarm.query(prompt)
            votes = {"YES": 0, "NO": 0, "HOLD": 0}
            details = []

            for provider, data in result["responses"].items():
                if not data["success"]:
                    continue
                text = data["response"]
                vote = "YES" if "YES" in text.upper() else "NO" if "NO" in text.upper() else "HOLD"
                votes[vote] += 1
                details.append(f"• {provider.upper()}: {vote}")

            agreement = max(votes.values())
            total = sum(votes.values())

            if agreement >= MIN_AGREEMENT_FOR_ALERT:
                side = max(votes, key=votes.get)
                msg = f"""
{agreement}/{total} MODELS AGREE – {'STRONG' if agreement >= 5 else 'HIGH'} {side}

Top fresh markets analyzed
Link: https://polymarket.com

Votes:
""" + "\n".join(details)

                self.send_telegram(msg)
                cprint(msg, "green", attrs=['bold'])

            self.last_analysis_run_timestamp = datetime.now().isoformat()

        except Exception as e:
            cprint(f"Swarm error: {e}", "red")

    def send_telegram(self, message: str):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        try:
            from telegram import Bot
            import asyncio
            async def send():
                bot = Bot(token=TELEGRAM_BOT_TOKEN)
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, disable_web_page_preview=True)
            asyncio.run(send())
            cprint("Telegram alert sent!", "green")
        except Exception as e:
            cprint(f"Telegram failed: {e}", "red")

# ==============================================================================
# Start the agent
# ==============================================================================

if __name__ == "__main__":
    agent = PolymarketAgent()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cprint("\nAgent stopped by user", "yellow")