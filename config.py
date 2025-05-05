from typing import Dict, TypedDict

class ModelPricing(TypedDict):
    input_price_per_1m: float
    output_price_per_1m: float

# Prices per 1 million tokens
MODEL_PRICING: Dict[str, ModelPricing] = {
    # ─────────────── OpenAI ───────────────
    "gpt-4-turbo-preview": {
        "input_price_per_1m": 10.0,
        "output_price_per_1m": 30.0,
    },
    "gpt-4": {
        "input_price_per_1m": 30.0,
        "output_price_per_1m": 60.0,
    },
    "gpt-3.5-turbo": {
        "input_price_per_1m": 1.0,
        "output_price_per_1m": 2.0,
    },
    "gpt-4o": {
        "input_price_per_1m": 5.0,
        "output_price_per_1m": 15.0,
    },
    "gpt-o3": {
        "input_price_per_1m": 10.0,
        "output_price_per_1m": 40.0,
    },
    "gpt-4.1": {
        "input_price_per_1m": 2.0,
        "output_price_per_1m": 8.0,
    },
    "gpt-4o-mini-high": {
        "input_price_per_1m": 0.15,
        "output_price_per_1m": 0.60,
    },

    # ────────────── Anthropic ─────────────
    "claude-3-opus-20240229": {
        "input_price_per_1m": 15.0,
        "output_price_per_1m": 75.0,
    },
    "claude-3-sonnet-20240229": {
        "input_price_per_1m": 3.0,
        "output_price_per_1m": 15.0,
    },
    "claude-3-haiku-20240307": {
        "input_price_per_1m": 0.25,
        "output_price_per_1m": 1.25,
    },

    # ─────────────── DeepSeek ─────────────
    "deepseek-chat": {
        "input_price_per_1m": 0.27,
        "output_price_per_1m": 1.10,
    },
    "deepseek-reasoner": {
        "input_price_per_1m": 0.55,
        "output_price_per_1m": 2.19,
    },

    # ─────────────── Google ───────────────
    "gemini-1.0-pro": {
        "input_price_per_1m": 0.25,
        "output_price_per_1m": 0.50,
    },
    "gemini-1.5-pro": {
        "input_price_per_1m": 0.50,
        "output_price_per_1m": 1.50,
    },
    "gemini-2.5-flash-preview-04-17": {
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.40,
    },
    "gemini-2.5-pro-exp-03-25": {
        "input_price_per_1m": 1.25,
        "output_price_per_1m": 10.0,
    },

    # ──────────────── Meta ────────────────
    "meta-llama/Llama-3.3-70B-Instruct": {
        "input_price_per_1m": 0.80,
        "output_price_per_1m": 0.88,
    },
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": {
        "input_price_per_1m": 0.27,
        "output_price_per_1m": 0.85,
    },

    # ─────────────── Alibaba ──────────────
    "qwen3-235b-a22b": {
        "input_price_per_1m": 0.20,
        "output_price_per_1m": 0.60,
    },

    # ──────────────── xAI ─────────────────
    "grok-3": {
        "input_price_per_1m": 3.0,
        "output_price_per_1m": 15.0,
    },
}


# Default pricing for unknown models
DEFAULT_PRICING: ModelPricing = {
    "input_price_per_1m": 1.0,
    "output_price_per_1m": 2.0
}

def get_model_pricing(model_name: str) -> ModelPricing:
    """Get pricing information for a specific model."""
    # Handle OpenRouter models by removing the prefix
    if model_name.startswith("openrouter/"):
        model_name = model_name.replace("openrouter/", "", 1)
    return MODEL_PRICING.get(model_name, DEFAULT_PRICING)

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the cost for a given model and token usage."""
    pricing = get_model_pricing(model_name)
    input_cost = (prompt_tokens * pricing["input_price_per_1m"]) / 1000000
    output_cost = (completion_tokens * pricing["output_price_per_1m"]) / 1000000
    return input_cost + output_cost 