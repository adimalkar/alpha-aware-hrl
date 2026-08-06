"""
LLM-based Analyst Agent for Regime Detection

Processes financial news and market context to output a "Regime Signal"
that guides the low-level trading agent.
"""

import re
import torch
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class RegimeSignal:
    """Output from the LLM Analyst."""
    regime: int  # 0=Safe, 1=Risky, 2=Crash
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Explanation for interpretability


class LLMAnalyst:
    """
    LLM-based analyst for market regime detection.
    
    Uses a small language model (TinyLlama or Phi-2) to process
    financial news and output regime signals for the trading agent.
    
    Args:
        model_name: HuggingFace model identifier
        device: torch device
        max_tokens: Maximum tokens for generation
    """
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "cuda",
        max_tokens: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
    def initialize(self):
        """Lazy initialization of the LLM model."""
        if self._initialized:
            return
            
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer and model. Use fp16 only when on CUDA.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if "cuda" in self.device else None,
        )
        
        # Ensure it is moved to the correct device manually if map="auto" didn't catch via accelerate
        if "cuda" in self.device and not hasattr(self.model, "hf_device_map"):
            self.model = self.model.to(self.device)
            
        self.model.eval()
        
        self._initialized = True
        print(f"[LLMAnalyst] Initialized with {self.model_name}")
        
    def analyze(
        self,
        news_articles: List[str],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> RegimeSignal:
        """
        Analyze news and market context to detect regime.
        
        Args:
            news_articles: List of recent financial news headlines/articles
            market_data: Optional market indicators (volatility, volume, etc.)
            
        Returns:
            RegimeSignal with regime classification and confidence
        """
        if not self._initialized:
            self.initialize()
        
        # Build prompt for regime classification
        prompt = self._build_prompt(news_articles, market_data)
        
        # Regex is imported at module level
        
        # Format the prompt using TinyLlama's chat template
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert financial analyst AI. "
                    "You must evaluate the news and strictly output only three lines exactly as shown below:\n"
                    "REGIME: 0\n"
                    "CONFIDENCE: 0.9\n"
                    "REASONING: The market is completely stable."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate response. Keep it short but long enough to include 3 lines.
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=min(self.max_tokens, 96),
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
        # Decode only the newly generated tokens
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Parse output utilizing Regex
        regime = 0       # Default: Safe
        confidence = 0.5 # Default Medium
        reasoning = "Failed to parse LLM Response"
        
        print(f"\n[DEBUG_LLM] Raw Response:\n{response_text}\n")
        
        try:
            regime_match = re.search(r"REGIME:\s*([0-2])", response_text, flags=re.IGNORECASE)
            conf_match = re.search(r"CONFIDENCE:\s*([0-1](?:\.\d+)?)", response_text, flags=re.IGNORECASE)
            reason_match = re.search(r"REASONING:\s*(.*)", response_text, flags=re.IGNORECASE)

            if regime_match:
                regime = int(regime_match.group(1))
            elif "CRASH" in response_text.upper() or "CIRCUIT BREAK" in response_text.upper():
                regime = 2
            elif "SAFE" in response_text.upper() or "NORMAL" in response_text.upper():
                regime = 0
            elif re.search(r'\bRISKY\b', response_text, re.IGNORECASE) or re.search(r'\bRISK\b', response_text, re.IGNORECASE):
                regime = 1
            else:
                regime = 1  # default to risky when uncertain

            if conf_match:
                confidence = float(conf_match.group(1))
                confidence = float(max(0.0, min(1.0, confidence)))

            if reason_match:
                reasoning = reason_match.group(1).strip()[:200]
            else:
                reasoning = response_text.replace("\n", " ").strip()[:200]

        except Exception as e:
            reasoning = f"Parse Error: {e} | Raw Output: {response_text}"
             
        return RegimeSignal(
            regime=regime,
            confidence=confidence,
            reasoning=reasoning,
        )
    
    def _build_prompt(
        self,
        news_articles: List[str],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the prompt for regime classification."""
        news_text = "\n".join([f"- {article}" for article in news_articles[:5]])
        
        prompt = f"""You are a financial analyst. Based on the following news and market data, classify the current market regime.

Recent News:
{news_text}

Market Indicators:
{market_data if market_data else "Not available"}

Classify the market regime as one of:
0 - SAFE: Normal market conditions, low volatility
1 - RISKY: Elevated uncertainty, moderate volatility
2 - CRASH: High risk, extreme volatility, potential crash

Respond with:
REGIME: [0/1/2]
CONFIDENCE: [0.0-1.0]
REASONING: [Brief explanation]
"""
        return prompt
    
    def get_regime_embedding(self, regime_signal: RegimeSignal) -> torch.Tensor:
        """Convert regime signal to embedding for RL agent."""
        # One-hot encoding of regime
        regime_onehot = torch.zeros(3)
        regime_onehot[regime_signal.regime] = 1.0
        
        # Add confidence as additional feature
        return torch.cat([regime_onehot, torch.tensor([regime_signal.confidence])])
