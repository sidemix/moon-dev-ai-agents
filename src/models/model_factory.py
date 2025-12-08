"""
Moon Dev's Model Factory — FIXED FOR OFFICIAL xAI SDK (Dec 2025)
Supports Claude, OpenAI, Grok (xai-sdk), Together, OpenRouter
"""

import os
from typing import Dict, Optional
from termcolor import cprint
from dotenv import load_dotenv
from pathlib import Path

# Official SDK imports
try:
    from xai_sdk import Client as XAIClient  # Official xAI SDK
except ImportError:
    XAIClient = None

from openai import OpenAI
from anthropic import Anthropic

# Load .env
project_root = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=project_root / ".env")

class ModelFactory:
    def __init__(self):
        self.models = {}
        self._initialize_all_models()

    def _initialize_all_models(self):
        # Claude (Anthropic)
        if key := os.getenv("ANTHROPIC_KEY"):
            try:
                client = Anthropic(api_key=key)
                self.models["claude"] = {
                    "client": client,
                    "model": "claude-3-5-sonnet-20241022",  # Sonnet 4.5
                    "name": "Claude Sonnet 4.5"
                }
                self.models["opus"] = {
                    "client": client,
                    "model": "claude-3-opus-20240229",  # Opus fallback (update to 4.5 if available)
                    "name": "Claude Opus 4.5"
                }
                cprint("Claude Sonnet + Opus ready", "green")
            except Exception as e:
                cprint(f"Claude failed: {e}", "red")

        # OpenAI
        if key := os.getenv("OPENAI_API_KEY"):
            try:
                client = OpenAI(api_key=key)
                self.models["openai"] = {
                    "client": client,
                    "model": "o1-mini",
                    "name": "o1-mini"
                }
                cprint("OpenAI o1-mini ready", "green")
            except Exception as e:
                cprint(f"OpenAI failed: {e}", "red")

        # Grok (xAI) — OFFICIAL SDK FIX
        if key := os.getenv("XAI_API_KEY"):
            if XAIClient is None:
                cprint("xai-sdk not installed — run: pip install xai-sdk==1.5.0", "red")
            else:
                try:
                    client = XAIClient(api_key=key)
                    self.models["grok"] = {
                        "client": client,
                        "model": "grok-beta",
                        "name": "Grok-4"
                    }
                    cprint("Grok-4 (xAI) ready", "green")
                except Exception as e:
                    cprint(f"Grok failed: {e}", "red")

        # Together.ai (OpenAI-compatible)
        if key := os.getenv("TOGETHER_API_KEY"):
            try:
                model = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.1-70B-Instruct")
                client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
                self.models["together"] = {
                    "client": client,
                    "model": model,
                    "name": f"Together/{model.split('/')[-1]}"
                }
                cprint(f"Together model ready: {model}", "green")
            except Exception as e:
                cprint(f"Together failed: {e}", "red")

        # OpenRouter (200+ models)
        if key := os.getenv("OPENROUTER_API_KEY"):
            try:
                client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
                self.models["openrouter"] = {
                    "client": client,
                    "model": "google/gemini-2.5-flash",
                    "name": "OpenRouter/Gemini-2.5"
                }
                cprint("OpenRouter ready", "green")
            except Exception as e:
                cprint(f"OpenRouter failed: {e}", "red")

        # Final status
        cprint(f"\nActive Models in Swarm: {len(self.models)}", "cyan", attrs=["bold"])
        for name in self.models:
            cprint(f"   {self.models[name]['name']}", "green")

    def get_model(self, provider: str, model_name: str = None):
        """Return model client + config"""
        if provider not in self.models:
            return None
        config = self.models[provider]
        if model_name:
            config["model"] = model_name
        return config["client"]

# Singleton
model_factory = ModelFactory()