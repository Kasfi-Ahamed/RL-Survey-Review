# SIT796 - RLHF Literature Review & Implementation

**Unit:** SIT796 Reinforcement Learning - Distinction Task 10.2  
**Survey reviewed:** Kaufmann, T., Weng, P., Bengs, V., & Hüllermeier, E. (2025). *A Survey of Reinforcement Learning from Human Feedback.* [`arXiv:2312.14925`](https://arxiv.org/abs/2312.14925)

---

## What this repo contains

| File | Description |
|------|-------------|
| `SIT796_DT10_2_RLHF_Review.ipynb` | Full implementation notebook - original diagrams, gridworld RLHF pipeline, and ablation experiments |

---

## What the notebook does

The notebook accompanies a written literature review that reconstructs the RLHF field as **one closed feedback loop with three design hinges** (feedback type → label collection → reward model), rather than a sequence of isolated topics.

It contains seven original figures and a working implementation:

- **Figure 1** - RLHF as a closed loop (synthesising Algorithm 1 + Figs 2-5 of the survey)
- **Figure 2** - The domino effect: how one feedback-type choice locks in every downstream decision
- **Figure 3** - Field timeline: PbRL (2011) → Deep RLHF (2017) → DPO (2023)
- **Figure 4** - Experimental results: learned reward vs ground-truth reward in a 5×5 gridworld
- **Figure 5** - Rationality sweep: performance as oracle irrationality (β) varies
- **Figure 6** - Sample efficiency: how many preferences are actually needed?
- **Figure 7** - Structural overview of the survey as a four-stage progression

### Implementation highlights

- 5×5 gridworld with a hidden lava column (col 2, R = −1) and goal cell (4,4, R = +10)
- Synthetic Bradley–Terry oracle with tunable rationality parameter β
- Reward model trained from 300 pairwise preferences using logistic NLL (Eq. 2 of the survey)
- Q-learning policy trained on the learned reward
- Result: **100% of oracle performance recovered** from preferences alone

---

## Requirements

```
numpy
matplotlib
```

No additional packages needed - all standard library otherwise.

---

## How to run

```bash
git clone https://github.com/Kasfi-Ahamed/RL-Survey-Review.git
cd RL-Survey-Review
jupyter notebook SIT796_DT10_2_RLHF_Review.ipynb
```

Run all cells top-to-bottom. Random seeds are fixed (`numpy` seed 42, `random` seed 42) so results are reproducible.

---

## Key finding

> Feedback type is an **architectural choice**, not a stylistic one. Choosing pairwise comparisons forces the Bradley–Terry model, logistic NLL loss, ensemble-variance queries, PPO/SAC, and Eluder-dimension regret bounds, the entire downstream stack follows from one Section 3 decision.
