#  Expected SARSA Agent - Taxi-v3 🚕

This project implements an **Expected SARSA reinforcement learning agent** to solve the classic `Taxi-v3` environment from OpenAI Gym.

---

## 📌 Environment Details

- **Environment:** `Taxi-v3`
- **Action Space:** 6 discrete actions
- **Observation Space:** 500 discrete states
- **Goal:** Pick up and drop off passengers in the correct locations with minimum time and penalties.

---

##  Algorithm Used

###  Expected SARSA

Expected SARSA is an on-policy Temporal Difference (TD) control algorithm. It calculates the expected value of the next state over the current policy (ε-greedy), making updates smoother and often more stable than standard Q-Learning or SARSA.

**Key Update Formula:**

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot \mathbb{E}_{a'}[Q(s', a')] - Q(s, a) \right]
\]

---

##  Performance

- ✅ **Average reward after training (20,000 episodes):** **9.2**
- ✅ Reached consistent success rate > 9.1 (OpenAI benchmark)
- 🔄 Epsilon-greedy exploration used with fixed ε = 0.001

---

## ⚙ Hyperparameters

| Parameter        | Value     |
|------------------|-----------|
| Learning rate (α) | 1.0       |
| Discount factor (γ) | 1.0     |
| Exploration rate (ε) | 0.001  |
| Episodes trained | 20,000    |

---

## 🛠 How to Run

### 1. Install dependencies

```bash
pip install gym numpy
