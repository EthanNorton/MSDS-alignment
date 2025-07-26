# 📈 Deepening Foundations for CMU MSML: From Stochastic Optimization to Temporal Reasoning

This document outlines key **concepts, articles, and growth areas** to master for a strong foundation in **deep learning**, **stochastic optimization**, **Markov Decision Processes**, and **temporal difference learning**, aligned with the research and applied focus of the [CMU MSML Program](https://ml.cmu.edu/academics/machine-learning-masters-curriculum).

---

## 🎯 Big Picture Alignment

CMU MSML emphasizes:
- Theoretical rigor in machine learning foundations
- Optimization, generalization, and probabilistic inference
- Applications in deep learning, RL, and scalable ML systems

To align with this vision, this roadmap builds on areas like:
- Advanced stochastic methods
- Reinforcement learning and dynamic programming
- Temporal learning and uncertainty quantification
- Convex/non-convex optimization

---

## 📚 Core Topics & Growth Areas

### 1. **Stochastic Optimization & Variants**
> *Focus*: How deep learning models are trained efficiently under uncertainty

- **Topics to Learn**
  - SGD, Adam, RMSprop, Nadam
  - Variance reduction: SVRG, SAG, SAGA
  - Generalization under noise

- **Growth Areas**
  - Adaptive vs non-adaptive learning
  - Large batch vs small batch behavior
  - Non-convex loss surface exploration
  - Deep ensembles and gradient variance tracking
  - Multi-seed evaluation as a stochastic stability benchmark

- **Recommended Reading**
  - 📄 [Understanding the Difficulty of Training Deep Feedforward Neural Networks (Glorot & Bengio, 2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
  - 📄 [The Marginal Value of Adaptive Gradient Methods in Machine Learning (Wilson et al., 2017)](https://arxiv.org/abs/1705.08292)

---

### 2. **Convex & Non-Convex Optimization**
> *Focus*: Theoretical grounding in optimization landscapes

- **Topics to Learn**
  - Convex sets, duality, KKT conditions
  - Strong vs weak convexity
  - Saddle points & escaping strategies

- **Recommended Courses**
  - Stanford CS109 - Probability [To be taken Fall 2025]. 
  - EE364a (Convex Optimization) by Stephen Boyd

- **Growth Areas**
  - Second-order methods vs first-order
  - Constraint-aware training
  - Optimization in high dimensions
  - Analysis of loss surface curvature and optimization plateaus

---

### 3. **Deep Learning & Generalization**
> *Focus*: Building stronger intuition for representation learning

- **Topics to Learn**
  - Initialization schemes & vanishing gradients
  - Overparameterization & double descent
  - BatchNorm, Dropout, LayerNorm

- **Growth Areas**
  - Implicit regularization in gradient descent
  - Deep ensemble learning for uncertainty calibration
  - Transformer inductive biases

- **Articles**
  - 📄 [Deep Learning Theory Review (Zeyuan Allen-Zhu)](https://arxiv.org/abs/2012.06291)

---

### 4. **Markov Decision Processes & Temporal Difference Learning**
> *Focus*: Foundations for Reinforcement Learning and sequential decision-making

- **Topics to Learn**
  - Bellman equations, policy/value functions
  - TD(λ), SARSA, Q-learning
  - Function approximation for RL

- **Recommended Books**
  - 📘 Sutton & Barto — Reinforcement Learning: An Introduction
  - 📘 David Silver’s RL Course (UCL): [Lecture Series](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

- **Growth Areas**
  - Offline RL and batch-constrained Q-learning (BCQ)
  - Exploration-exploitation trade-offs
  - Temporal abstraction and options

**🔍 Connection to Paper:**  
Consider modeling **grid interaction as a decision-making problem**—i.e., how pricing or consumption incentives could be framed as an MDP. This sets the stage for your RL expansion in future work.

---

### 5. **Temporal Modeling & Sequential Learning**
> *Focus*: Capturing time-dependent patterns and uncertainty

- **Topics**
  - RNNs, LSTMs, GRUs
  - Temporal convolutions (TCNs)
  - Attention over time, transformers for time series

- **Projects**
  - Energy forecasting with LSTM + seasonal weighting (already started)
  - TCN vs LSTM comparison
  - Bayesian RNNs (MC Dropout)
  - Use of attention weights to analyze peak-demand periods


---

## 🔁 Integration & Next Steps

| Phase | Focus | Action Items |
|-------|-------|--------------|
| **Now** | Stabilize stochastic training | Run Optuna-based tuning across seeds; compare Adam vs SGD |
| **Next 1–2 months** | Theory & reproducibility | Work through CS109 + replicate loss surface studies |
| **Fall** | Reinforcement Learning core | Finish Sutton & Barto, implement SARSA / TD(λ) |
| **Ongoing** | Research alignment | Follow CMU MSML faculty publications on RL, optimization, and theory |
| **Optional** | Generalization analysis | Explore overparameterization, implicit bias, and double descent |

---

## 📂 GitHub Integration

Projects and study areas to be added: 
- `stochastic_optimization/` – with seed tuning + optimizer experiments
- `deep_rl_td_learning/` – with SARSA and TD(λ) baselines
- `temporal_learning/` – with Fourier LSTM, TCNs, and MC Dropout models

Each project folder should contain:
- `notebooks/` with annotated training pipelines
- `notes/` with theoretical interpretations
- `plots/` for reproducibility and variance tracking
