#!/usr/bin/env python3
"""
Test Script for LLM Analyst (Step 6)

This script validates that the `LLMAnalyst` successfully loads
the TinyLlama checkpoint and acts as a proper instruction-tuned oracle
under expected news flow, returning parsed Regime Signals.
"""

import sys
sys.path.insert(0, '.')

import torch
from src.agents.llm_analyst import LLMAnalyst

def main():
    print("="*70)
    print("Testing LLM Analyst Feature (Step 6)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Initialize Wrapper
    print("\nInitializing LLM Analyst...")
    analyst = LLMAnalyst(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=device,
        max_tokens=256
    )
    
    # 2. Create Mock Scenarios
    scenarios = [
        {
            "name": "Normal Market Day",
            "news": [
                "Tech giants hold steady amidst minor earnings reports.",
                "Federal reserve signals no rate hikes are expected this quarter.",
                "Unemployment figures match expectations across the board."
            ],
            "market": {"VIX": 14.5, "Volume": "Average"}
        },
        {
            "name": "High Risk Event",
            "news": [
                "Major semiconductor manufacturer reports critical supply chain disruptions.",
                "Investors express severe anxiety heading into tomorrow's inflation data release.",
                "Unidentified trading firm liquidates $500M in positions unexpectedly."
            ],
            "market": {"VIX": 25.1, "Volume": "High"}
        },
        {
            "name": "Crash Condition",
            "news": [
                "BREAKING: Largest global bank declares bankruptcy.",
                "S&P 500 futures trigger circuit breakers down 6% premarket.",
                "Widespread panic selling observed across all major sectors."
            ],
            "market": {"VIX": 65.0, "Volume": "Extreme"}
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- Testing Scenario: {scenario['name']} ---")
        try:
            signal = analyst.analyze(
                news_articles=scenario['news'],
                market_data=scenario['market']
            )
            print(f"  Regime: {signal.regime} (0=Safe, 1=Risky, 2=Crash)")
            print(f"  Confidence: {signal.confidence}")
            print(f"  Reasoning: {signal.reasoning}")
            
            # Map test expected output: Crash should absolutely not be 0.
            if scenario['name'] == 'Crash Condition':
                assert signal.regime == 2, "Failed to identify clear crash scenario!"
                
            embedding = analyst.get_regime_embedding(signal)
            print(f"  Resulting Vector Embedding: {embedding.tolist()}")
            
        except Exception as e:
            print(f"  ❌ LLM Analysis Failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    print("\n" + "="*70)
    print("All LLM Analyst tests completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
