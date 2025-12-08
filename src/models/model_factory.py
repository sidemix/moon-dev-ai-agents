# src/models/model_factory.py
import os
from openai import OpenAI
from anthropic import Anthropic
from xai_sdk import Client as XAIClient
from termcolor import cprint

class ModelFactory:
    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        # Claude
        if key := os.getenv("ANTHROPIC_KEY"):
            try:
                client = Anthropic(api_key=key)
                self.models["claude"] = {"client": client, "model": "claude-3-5-sonnet-20241022", "name": "Claude Sonnet 4.5"}
                self.models["opus"] = {"client": client, "model": "claude-3-opus-20240229", "name": "Claude Opus 4.5"}
                cprint("Claude Sonnet + Opus ready", "green")
            except: pass

        # OpenAI
        if key := os.getenv("OPENAI_API_KEY"):
            try:
                client = OpenAI(api_key=key)
                self.models["openai"] = {"client": client, "model": "o1-mini", "name": "o1-mini"}
                cprint("OpenAI o1-mini ready", "green")
            except: pass

        # Grok (xAI)
        if key := os.getenv("XAI_API_KEY"):
            try:
                client = XAIClient(api_key=key)
                self.models["grok"] = {"client": client, "model": "grok-beta", "name": "Grok-4"}
                cprint("Grok-4 (xAI) ready", "green")
            except: pass

        # Together.ai
        if key := os.getenv("TOGETHER_API_KEY"):
            try:
                model = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
                client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
                self.models["together"] = {"client": client, "model": model, "name": "Together/Llama-3.3-70B"}
                cprint(f"Together model ready: {model}", "green")
            except: pass

        # OpenRouter
        if key := os.getenv("OPENROUTER_API_KEY"):
            try:
                client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
                self.models["openrouter"] = {"client": client, "model": "google/gemini-2.5-flash", "name": "OpenRouter/Gemini-2.5"}
                cprint("OpenRouter ready", "green")
            except: pass

        # DEEPSEEK — THIS IS THE NEW ADDITION
        if key := os.getenv("DEEPSEEK_KEY"):
            try:
                model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
                client = OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
                self.models["deepseek"] = {"client": client, "model": model, "name": f"DeepSeek/{model}"}
                cprint(f"DeepSeek ready: {model}", "green")
            except Exception as e:
                cprint(f"DeepSeek failed: {e}", "red")

        cprint(f"\nActive Models in Swarm: {len(self.models)}", "cyan", attrs=["bold"])
        for name in self.models:
            cprint(f"   {self.models[name]['name']}", "green")

    def get_model(self, provider: str):
        return self.models.get(provider, {}).get("client")

    def get_config(self, provider: str):
        return self.models.get(provider, {})

model_factory = ModelFactory()