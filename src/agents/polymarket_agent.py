#!/usr/bin/env python3
"""
Moon Dev's Polymarket Agent — FINAL 6-MODEL + TELEGRAM VERSION
100% crash-proof — works perfectly on Railway
"""

import os
import sys
import time
import json
import pandas as pd
import threading
import websocket
from datetime import datetime, timedelta
from pathlib import Path
from termcolor import cprint

# Add project root
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.model_factory import model_factory

# ==============================================================================
# CONFIG
# ==============================================================================

MIN_TRADE_SIZE_USD = int(os.getenv("MIN_TRADE_SIZE_USD", "500"))
ANALYSIS_INTERVAL = int(os.getenv("ANALYSIS_INTERVAL", "300"))
NEW_MARKETS_FOR_ANALYSIS = int(os.getenv("NEW_MARKETS_FOR_ANALYSIS", "3"))
MIN_AGREEMENT_FOR_ALERT = int(os.getenv("MIN_AGREEMENT_FOR_ALERT", "4"))

WEBSOCKET_URL = "wss://ws-live-data.polymarket.com"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

DATA_FOLDER = Path(project_root) / "src/data/polymarket"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
MARKETS_CSV = DATA_FOLDER / "markets.csv"

# ==============================================================================
# Polymarket Agent — BULLETPROOF FINAL VERSION
# ==============================================================================

class PolymarketAgent:
    def __init__(self):
        cprint("\n" + "="*80, "cyan")
        cprint("Moon Dev's Polymarket Agent - 6-Model Swarm Edition", "cyan", attrs=['bold'])
        cprint("="*80, "cyan")

        self.csv_lock = threading.Lock()
        self.last_analysis_run = None
        self.ws_connected = False

        self.markets_df = self._load_csv()

        cprint(f"Loaded {len(self.markets_df)} markets", "green")

        self.connect_websocket()
        threading.Thread(target=self.status_loop, daemon=True).start()
        threading.Thread(target=self.analysis_loop, daemon=True).start()

        cprint("All systems GO!\n", "green", attrs=['bold'])

    def _load_csv(self):
        if MARKETS_CSV.exists():
            try:
                df = pd.read_csv(MARKETS_CSV)
                for col in ["market_id", "title", "event_slug", "last_trade"]:
                    if col not in df.columns:
                        df[col] = None
                return df
            except:
                pass
        return pd.DataFrame(columns=["market_id", "title", "event_slug", "last_trade"])

    def _save_csv(self):
        with self.csv_lock:
            self.markets_df.to_csv(MARKETS_CSV, index=False)

    def connect_websocket(self):
        cprint(f"Connecting to {WEBSOCKET_URL}...", "cyan")
        self.ws = websocket.WebSocketApp(
            WEBSOCKET_URL,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=lambda ws, e: cprint(f"WS Error: {e}", "red"),
            on_close=lambda *a: (time.sleep(5), self.connect_websocket())
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def on_open(self, ws):
        cprint("WebSocket Connected!", "green")
        self.ws_connected = True
        ws.send(json.dumps({"action": "subscribe", "subscriptions": [{"topic": "activity", "type": "orders_matched"}]}))

    def on_message(self, ws, msg):
        try:
            data = json.loads(msg)
            if data.get("type") != "orders_matched": return
            p = data.get("payload", {})
            usd = float(p.get("size", 0)) * float(p.get("price", 0))
            if usd < MIN_TRADE_SIZE_USD: return

            mid = p.get("conditionId")
            title = p.get("title", "Unknown")
            slug = p.get("eventSlug", "")
            if not mid or not slug: return

            if mid in self.markets_df["market_id"].values:
                self.markets_df.loc[self.markets_df["market_id"] == mid, "last_trade"] = datetime.now().isoformat()
            else:
                new_row = pd.DataFrame([{
                    "market_id": mid,
                    "title": title,
                    "event_slug": slug,
                    "last_trade": datetime.now().isoformat()
                }])
                self.markets_df = pd.concat([self.markets_df, new_row], ignore_index=True)
                cprint(f"NEW: ${usd:,.0f} → {title[:70]}", "green")

            self._save_csv()
        except: pass

    def status_loop(self):
        while True:
            time.sleep(30)
            df = self.markets_df.copy()
            fresh = len(df[df["last_trade"].isna() | (pd.to_datetime(df["last_trade"], errors="coerce") > (datetime.now() - timedelta(hours=8)))])
            cprint(f"\nStatus @ {datetime.now().strftime('%H:%M:%S')} | Markets: {len(df)} | Fresh: {fresh}", "cyan")

    def analysis_loop(self):
        cprint("First analysis starting NOW...", "yellow", attrs=['bold'])
        while True:
            self.run_analysis()
            time.sleep(ANALYSIS_INTERVAL)

    def run_analysis(self):
        df = self.markets_df.copy()

        # Safe fresh count and mask
        fresh_mask = pd.Series([True] * len(df), index=df.index)
        fresh_count = len(df)

        try:
            last_trade_times = pd.to_datetime(df["last_trade"], errors="coerce")
            cutoff = datetime.now() - timedelta(hours=8) if self.last_analysis_run else None
            fresh_mask = df["last_trade"].isna() | (last_trade_times > cutoff)
            fresh_count = fresh_mask.sum()
        except:
            pass

        if fresh_count < NEW_MARKETS_FOR_ANALYSIS and self.last_analysis_run:
            return

        cprint(f"\n6-MODEL ANALYSIS on {fresh_count} fresh markets!", "magenta", attrs=['bold'])

        # Fixed: markets.iterrows() returns (index, row), so use row directly
        markets_to_show = df[fresh_mask].tail(10) if fresh_count > 0 else df.tail(10)
        prompt_lines = []
        for i, (_, row) in enumerate(markets_to_show.iterrows(), 1):
            title = row["title"] if pd.notna(row["title"]) else "Unknown"
            slug = row["event_slug"] if pd.notna(row["event_slug"]) else ""
            prompt_lines.append(f"{i}. {title}\nhttps://polymarket.com/event/{slug}")

        prompt = "Analyze these Polymarket markets. Answer only YES, NO, or HOLD for each:\n\n" + "\n".join(prompt_lines)

        try:
            result = model_factory.swarm.query(prompt)
            votes = {"YES": 0, "NO": 0, "HOLD": 0}
            details = []

            for prov, data in result["responses"].items():
                if not data["success"]: continue
                text = data["response"]
                vote = "YES" if "YES" in text.upper() else "NO" if "NO" in text.upper() else "HOLD"
                votes[vote] += 1
                details.append(f"• {prov.upper()}: {vote}")

            agreement = max(votes.values())
            total = sum(votes.values())

            if agreement >= MIN_AGREEMENT_FOR_ALERT:
                side = max(votes, key=votes.get)
                msg = f"""
{agreement}/{total} MODELS AGREE – {'STRONG' if agreement >= 5 else 'HIGH'} {side}

Fresh markets analyzed
https://polymarket.com

Votes:
""" + "\n".join(details)

                self.send_telegram(msg)
                cprint(msg, "green", attrs=['bold'])

            self.last_analysis_run = datetime.now().isoformat()

        except Exception as e:
            cprint(f"Swarm error: {e}", "red")

    def send_telegram(self, msg):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
        try:
            from telegram import Bot
            import asyncio
            async def send():
                bot = Bot(token=TELEGRAM_BOT_TOKEN)
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, disable_web_page_preview=True)
            asyncio.run(send())
            cprint("Telegram alert sent!", "green")
        except Exception as e:
            cprint(f"Telegram failed: {e}", "red")

if __name__ == "__main__":
    agent = PolymarketAgent()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        cprint("\nAgent stopped", "yellow")