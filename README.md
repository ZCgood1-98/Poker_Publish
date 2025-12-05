# Evaluating Large Language Model’s performance with Poker
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/ishikota/kyoka/blob/master/LICENSE.md)

A comprehensive study evaluating the probabilistic reasoning and decision-making capabilities of leading Large Language Models (LLMs) using Texas Hold'em poker as a benchmark.

# Overview
This research addresses critical gaps in current LLM evaluation methodologies by testing models in scenarios requiring probabilistic reasoning, risk assessment, and strategic decision-making under uncertainty. Unlike traditional benchmarks that focus on deterministic games, this study uses poker to evaluate real-world decision-making capabilities.
  
# Models Evaluated
- Llama 4 Maverick
- Claude 3.5 Haiku
- GPT-4o-mini
- DeepSeek V3 0324
- Monte Carlo Tree Search algorithm as Baseline

# Key Findings
1. **MCTS Algorithm**: Had the highest chips at the end with 216,590 chips accumulated.
2. **DeepSeek V3**: Adopted an ultra-conservative strategy and ended up with the second highest chip earnings with 67,369 chips 
3. **Claude 3.5 Haiku**: Inconsistent performance throughout the experiemnt ending with 50,724 chips.
4. **Llama 4 Maverick**: Had the highest win rate with an overly aggressive behaviour but total chips accumulated was -154,612 chips
5. **GPT-4o-mini**: Poorest performance out of all the LLMs with -180,092 chips accumulated

# Installation

```python
git clone https://github.com/a1886375/Topics-in-CS-adv
cd dev
pip install -r requirements.txt
```

# Usage
```python
# Run game
python run_game.py

# Create/Update Visualisations
python create_vis.py llm_poker_evaluation_20250602.json
```

# PyPokerEngine Documentation
For mode detail, please checkout [doc site](https://ishikota.github.io/PyPokerEngine/)

# Acknowledgments
- Supervisor: Dr. Xinyu Wang \n
- PyPokerEngine for the poker simulation framework
- OpenRouter API for LLM acces

