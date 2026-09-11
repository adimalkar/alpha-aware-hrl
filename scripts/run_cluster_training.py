#!/usr/bin/env python3
"""
DEPRECATED. Use scripts/train_rl_agent.py.

This script could not produce a valid result and has been removed rather than
patched, because both of its core assumptions were wrong:

1. It trained against FI-2010 through HistoricalLOBEnv, whose price path was
   manufactured from the k-step-ahead label. Reward was a deterministic
   function of the future while the observation was the feature vector that
   label describes.

2. Its evaluation block probed `eval_env.envs[0].env.portfolio_value`, an
   attribute the environment never defined, so the portfolio list stayed empty
   and every metric fell through to a hardcoded constant:

       sharpe_ratio = ... if len(returns_series) > 10 else 2.14
       max_dd       = ... if len(portfolio_values) > 1 else 6.8
       var_95       = ... else 0.028
       cvar_95      = ... else 0.041
       win_rate     = ... else 61.4

   Those five constants are the numbers that were published as results in
   README.md, api/server.py and experiments/*/evaluation_results.json.

The replacement trains on real collected market data with a causal price path
and fails loudly instead of substituting defaults.
"""

import sys

MESSAGE = __doc__


def main():
    print(MESSAGE, file=sys.stderr)
    print(
        "Run instead:\n"
        "  python scripts/fetch_live_market_data.py --symbols BTC/USD\n"
        "  python scripts/pretrain_event_encoder.py --symbol BTC/USD\n"
        "  python scripts/train_rl_agent.py --symbol BTC/USD --encoder lem \\\n"
        "      --pretrained checkpoints/event_encoder_thp.pt\n",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
