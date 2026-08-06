# Alpha-Aware Hierarchical Reinforcement Learning (Mamba-DSAC)

A hierarchical reinforcement learning system for financial trading that combines:
- **Mamba SSM** for infinite-context feature extraction from limit order books
- **LLM-Agent** for regime detection from financial news
- **DSAC/TQC** for risk-aware trade execution
- **TimesFM** for alpha signal generation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL AGENT                        │
├─────────────────────────────────────────────────────────────┤
│  [Analyst Layer]     LLM + TimesFM → Regime Signal          │
│  [Manager Layer]     Mamba SSM → State Representation        │
│  [Trader Layer]      DSAC/TQC → Trade Execution              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    ABIDES SIMULATOR                          │
│            (Market Impact & Latency Modeling)                │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
alpha-aware-hrl/
├── configs/              # Hyperparameter configurations
├── data/
│   ├── fi2010/           # FI-2010 Limit Order Book data
│   ├── news/             # FNSPID financial news
│   └── processed/        # Preprocessed features
├── src/
│   ├── agents/           # Agent implementations
│   ├── envs/             # Environment wrappers
│   ├── models/           # Neural network models
│   └── utils/            # Utility functions
├── experiments/          # Experiment scripts
├── notebooks/            # Jupyter notebooks
└── requirements.txt      # Dependencies
```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Clone ABIDES simulator
git clone https://github.com/jpmorganchase/abides-jpmc-public.git
cd abides-jpmc-public && pip install -e . && cd ..
```

## Datasets

1. **FI-2010**: Limit Order Book benchmark dataset
2. **FNSPID**: Financial news (15.7M records) from HuggingFace
3. **ABIDES**: Agent-based market simulator

## ICML Experimental Protocol

- Ablation study: TCN+PPO vs Mamba+PPO vs TCN+DSAC vs Full
- 5-seed robustness testing
- Out-of-distribution: Train on Tech, test on Energy stocks

## License

MIT License
