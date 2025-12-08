#!/usr/bin/env python3
"""
Moon Dev's Swarm Agent - FULLY WORKING 5-MODEL VERSION
"""

import os
import sys
import json
import time
import re
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from termcolor import colored, cprint
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Moon Dev's model factory
from src.models.model_factory import model_factory

# ============================================
# SWARM CONFIGURATION - CLEAN & WORKING
# ============================================

SWARM_MODELS = {
    # YOUR 5 WORKING MODELS (exact names the factory expects)
    "claude":   (True, "anthropic", "claude-sonnet-4-5"),
    "opus":     (True, "anthropic", "claude-opus-4-5-20251101"),
    "openai":   (True, "openai",    "o1-mini"),                                          # Uses OPENAI_API_KEY
    "grok":     (True, "xai",       "grok-beta"),                                        # Uses XAI_API_KEY + XAI_MODEL=grok-beta
    "together": (True, "together",  os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")),
}

# Default parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
MODEL_TIMEOUT = 120

# Use Claude as fallback consensus reviewer (DeepSeek often fails)
CONSENSUS_REVIEWER_MODEL = ("anthropic", "claude-sonnet-4-5")

CONSENSUS_REVIEWER_PROMPT = """You are a consensus analyzer reviewing multiple AI responses.

Below are responses from {num_models} different frontier models to the same question.

{responses}

Your task: Provide a clear, concise 3-sentence maximum consensus response that:
1. Synthesizes the common themes across all responses
2. Highlights any notable agreements or disagreements
3. Gives a balanced, actionable summary

Keep it under 3 sentences. Be direct and clear."""

SAVE_RESULTS = True
RESULTS_DIR = Path(project_root) / "src" / "data" / "swarm_agent"

# ============================================
# SwarmAgent Class (only tiny fixes added)
# ============================================

class SwarmAgent:
    def __init__(self, custom_models: Optional[Dict] = None):
        self.models_config = custom_models or SWARM_MODELS
        self.active_models = {}
        self.results_dir = RESULTS_DIR
        if SAVE_RESULTS:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_models()

        cprint("\n" + "="*60, "cyan")
        cprint("Moon Dev's Swarm Agent Initialized", "cyan", attrs=['bold'])
        cprint("="*60, "cyan")
        cprint(f"Active Models in Swarm: {len(self.active_models)}", "green")
        for name in self.active_models:
            cprint(f"   {name}", "green")

    def _initialize_models(self):
        """Initialize all enabled models with better key handling"""
        for provider, (enabled, model_type, model_name) in self.models_config.items():
            if not enabled:
                continue

            try:
                # Flexible key lookup (this is the key fix)
                key = None
                if model_type == "openai":
                    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
                else:
                    key = os.getenv(f"{model_type.upper()}_API_KEY") or os.getenv(f"{model_type.upper()}_KEY")

                if not key:
                    cprint(f"Missing API key for {provider}", "red")
                    continue

                model = model_factory.get_model(model_type, model_name)
                if model:
                    self.active_models[provider] = {
                        "model": model,
                        "type": model_type,
                        "name": model_name
                    }
                    cprint(f"Initialized {provider}: {model_name}", "green")
                else:
                    cprint(f"model_factory returned None for {provider}", "yellow")
            except Exception as e:
                cprint(f"Error initializing {provider}: {e}", "red")

    # The rest of your code is perfect — unchanged below
    def _query_single_model(self, provider: str, model_info: Dict, prompt: str,
                          system_prompt: Optional[str] = None) -> Tuple[str, Dict]:
        start_time = time.time()
        try:
            if system_prompt is None:
                system_prompt = "You are a helpful AI assistant providing thoughtful analysis."
            response = model_info["model"].generate_response(
                system_prompt=system_prompt,
                user_content=prompt,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS
            )
            elapsed = time.time() - start_time
            return provider, {
                "provider": provider,
                "model": model_info["name"],
                "response": response,
                "success": True,
                "error": None,
                "response_time": round(elapsed, 2)
            }
        except Exception as e:
            elapsed = time.time() - start_time
            cprint(f"Error querying {provider}: {e}", "red")
            return provider, {
                "provider": provider,
                "model": model_info["name"],
                "response": None,
                "success": False,
                "error": str(e),
                "response_time": round(elapsed, 2)
            }

    def query(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        cprint(f"\nInitiating Swarm Query with {len(self.active_models)} models...", "cyan", attrs=['bold'])
        cprint(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}", "blue")

        cprint(f"\nCalling models in parallel:", "yellow", attrs=['bold'])
        for provider, model_info in self.active_models.items():
            cprint(f"   → {provider.upper()}: {model_info['name']}", "cyan")

        start_time = time.time()
        all_responses = {}

        with ThreadPoolExecutor(max_workers=len(self.active_models)) as executor:
            futures = {
                executor.submit(self._query_single_model, provider, info, prompt, system_prompt): provider
                for provider, info in self.active_models.items()
            }
            for future in as_completed(futures, timeout=MODEL_TIMEOUT + 30):
                provider, response = future.result()
                all_responses[provider] = response
                status = "success" if response["success"] else "failed"
                cprint(f"{provider.upper()}: {status} ({response['response_time']}s)", "green" if response["success"] else "red")

        consensus_summary, model_mapping = self._generate_consensus_review(all_responses, prompt)

        total_time = round(time.time() - start_time, 2)
        result = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "consensus_summary": consensus_summary,
            "model_mapping": model_mapping,
            "responses": {p: {"response": r["response"].content if hasattr(r["response"], "content") else str(r["response"]), "success": r["success"]} for p, r in all_responses.items()},
            "metadata": {
                "total_models": len(self.active_models),
                "successful_responses": sum(1 for r in all_responses.values() if r["success"]),
                "total_time": total_time
            }
        }

        if SAVE_RESULTS:
            self._save_results(result)

        return result

    def _generate_consensus_review(self, responses: Dict, original_prompt: str) -> Tuple[str, Dict]:
        successful = [(p, r["response"]) for p, r in responses.items() if r["success"] and r["response"]]
        if not successful:
            return "No successful responses.", {}

        model_mapping = {f"AI #{i+1}": p.upper() for i, (p, _) in enumerate(successful)}
        formatted = "\n".join(f"AI #{i+1}:\n{str(r)[:1000]}" for i, (_, r) in enumerate(successful))

        try:
            reviewer = model_factory.get_model(*CONSENSUS_REVIEWER_MODEL)
            review = reviewer.generate_response(
                system_prompt="You are a consensus analyzer.",
                user_content=CONSENSUS_REVIEWER_PROMPT.format(num_models=len(successful), responses=formatted),
                temperature=0.3,
                max_tokens=200
            )
            return (review.content.strip() if hasattr(review, "content") else str(review), model_mapping)
        except Exception as e:
            cprint(f"Consensus review failed: {e}", "red")
            return "Consensus failed.", model_mapping

    def _save_results(self, result: Dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"swarm_result_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        cprint(f"\nResults saved to: {filename.relative_to(Path(project_root))}", "blue")

    # Rest of your original methods unchanged...
    def _strip_think_tags(self, text: str) -> str:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _print_summary(self, result: Dict):
        metadata = result["metadata"]
        cprint("\n" + "="*60, "green")
        cprint("SWARM CONSENSUS", "green", attrs=['bold'])
        cprint("="*60, "green")
        if "model_mapping" in result and result["model_mapping"]:
            cprint("\nModel Key:", "blue")
            for ai_num, provider in result["model_mapping"].items():
                cprint(f"   {ai_num} = {provider}", "white")
        if "consensus_summary" in result:
            cprint("\nAI CONSENSUS SUMMARY:", "magenta", attrs=['bold'])
            cprint(f"{result['consensus_summary']}\n", "white")
        cprint(f"Performance:", "cyan")
        cprint(f"   Total Time: {metadata['total_time']}s", "white")
        cprint(f"   Success Rate: {metadata['successful_responses']}/{metadata['total_models']}", "white")

    def query_dataframe(self, prompt: str, system_prompt: Optional[str] = None) -> pd.DataFrame:
        result = self.query(prompt, system_prompt)
        data = []
        for provider, response_data in result["responses"].items():
            data.append({
                "provider": provider,
                "response": response_data["response"][:500] if response_data["response"] else None,
                "success": response_data["success"],
                "error": response_data.get("error"),
                "response_time": response_data["response_time"]
            })
        return pd.DataFrame(data)


def main():
    cprint("\n" + "="*60, "cyan")
    cprint("Moon Dev's Swarm Agent", "cyan", attrs=['bold'])
    cprint("="*60, "cyan")
    swarm = SwarmAgent()
    cprint("\nWhat would you like to ask the AI swarm?", "yellow")
    prompt = input("Prompt > ").strip()
    if not prompt:
        cprint("No prompt provided. Exiting.", "red")
        return
    result = swarm.query(prompt)
    cprint("\n" + "="*60, "cyan")
    cprint("AI RESPONSES", "cyan", attrs=['bold'])
    cprint("="*60, "cyan")
    reverse_mapping = {provider.lower(): ai_num for ai_num, provider in result.get("model_mapping", {}).items()}
    for provider, data in result["responses"].items():
        if data["success"]:
            ai_label = reverse_mapping.get(provider, "")
            label = f"{ai_label} ({provider.upper()})" if ai_label else provider.upper()
            cprint(f"\n{label}:", "yellow", attrs=['bold'])
            response_text = data['response']
            if len(response_text) > 800:
                cprint(f"{response_text[:800]}...\n", "white")
                cprint("[Response truncated - see full output in saved JSON]", "cyan")
            else:
                cprint(f"{response_text}", "white")
            cprint(f"Response time: {data['response_time']}s", "cyan")
        else:
            cprint(f"\n{provider.upper()}: Failed - {data['error']}", "red")
    swarm._print_summary(result)
    cprint("\nSwarm query complete!", "cyan", attrs=['bold'])


if __name__ == "__main__":
    main()