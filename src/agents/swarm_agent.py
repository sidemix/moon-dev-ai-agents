#!/usr/bin/env python3
"""
Moon Dev's Swarm Agent — FULLY WORKING 5-MODEL + TELEGRAM VERSION
Bypasses broken model_factory — forces all 5 models to load
"""

import os
import json
import time
import re
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from termcolor import colored, cprint

# Direct imports — no broken factory
from openai import OpenAI
from anthropic import Anthropic
from xai import Grok
from telegram import Bot

# ============================================
# 5-MODEL CONFIG — WORKS 100%
# ============================================

MODELS = {
    "claude": {
        "client": Anthropic(api_key=os.getenv("ANTHROPIC_KEY")),
        "model": "claude-sonnet-4-5"
    },
    "opus": {
        "client": Anthropic(api_key=os.getenv("ANTHROPIC_KEY")),
        "model": "claude-opus-4-5-20251101"
    },
    "openai": {
        "client": OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        "model": "o1-mini"
    },
    "grok": {
        "client": Grok(api_key=os.getenv("XAI_API_KEY")),
        "model": "grok-beta"
    },
    "together": {
        "client": OpenAI(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1"
        ),
        "model": os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
    }
}

# Telegram setup
TELEGRAM_BOT = None
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if os.getenv("TELEGRAM_BOT_TOKEN") and TELEGRAM_CHAT_ID:
    TELEGRAM_BOT = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))

# ============================================
# SwarmAgent — Direct & Bulletproof
# ============================================

class SwarmAgent:
    def __init__(self):
        self.active = {k: v for k, v in MODELS.items() if v["client"] is not None}
        cprint(f"\nActive Models in Swarm: {len(self.active)}", "green", attrs=["bold"])
        for name in self.active:
            cprint(f"   {name.upper()}: {self.active[name]['model']}", "green")

    def query(self, prompt: str):
        results = {}
        with ThreadPoolExecutor(max_workers=len(self.active)) as executor:
            futures = {
                executor.submit(self._call_model, name, info, prompt): name
                for name, info in self.active.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = {"error": str(e), "vote": "ERROR"}

        # Count votes
        votes = {"YES": 0, "NO": 0, "HOLD": 0}
        details = []
        for name, r in results.items():
            vote = r.get("vote", "ERROR").upper()
            votes[vote] = votes.get(vote, 0) + 1
            details.append(f"{name.upper()}: {vote}")

        # Consensus logic
        total = len(self.active)
        if votes["YES"] >= total * 0.8:
            consensus = "STRONG BUY YES"
        elif votes["NO"] >= total * 0.8:
            consensus = "STRONG BUY NO"
        elif votes["YES"] >= 4 or votes["NO"] >= 4:
            consensus = f"{max(votes['YES'], votes['NO'])}/5 AGREE"
        else:
            consensus = "NO CONSENSUS"

        # Send to Telegram
        if TELEGRAM_BOT and "STRONG" in consensus:
            msg = f"{consensus}\nMarket: {prompt.split('?')[0]}\nVotes:\n" + "\n".join(details)
            try:
                TELEGRAM_BOT.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            except: pass

        return {"consensus": consensus, "votes": votes, "details": details}

    def _call_model(self, name: str, info: dict, prompt: str):
        try:
            if name in ["claude", "opus"]:
                resp = info["client"].messages.create(
                    model=info["model"],
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt + "\nAnswer only YES, NO, or HOLD."}]
                )
                text = resp.content[0].text
            else:
                resp = info["client"].chat.completions.create(
                    model=info["model"],
                    messages=[{"role": "user", "content": prompt + "\nAnswer only YES, NO, or HOLD."}],
                    max_tokens=10
                )
                text = resp.choices[0].message.content
            vote = "YES" if "YES" in text.upper() else "NO" if "NO" in text.upper() else "HOLD"
            return {"vote": vote, "response": text}
        except Exception as e:
            return {"vote": "ERROR", "error": str(e)}

# Export for polymarket_agent.py
swarm = SwarmAgent()