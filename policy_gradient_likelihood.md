# Policy Gradients: The Likelihood Perspective

## Introduction

Policy Gradient methods are fundamentally connected to **Maximum Likelihood Estimation (MLE)**. Understanding this connection provides deep insight into why these algorithms work and how they relate to classical statistical methods.

---

## 1. Likelihood in Policy Gradients

### What is Likelihood?

In statistics, we distinguish between probability and likelihood:

- **Probability** $P(\text{data} | \text{parameters})$: Given fixed parameters, what's the probability of observing this data?
- **Likelihood** $L(\text{parameters} | \text{data})$: Given observed data, how likely are these parameters?

**Key insight**: They are the **same mathematical function**, just interpreted from different perspectives!

### P(τ; θ) as a Likelihood Function

In Policy Gradients:

$$P(\tau; \theta) = \text{probability of trajectory } \tau \text{ occurring under policy with parameters } \theta$$

This can be interpreted in two ways:

- **As probability**: "Given my current policy $\pi_\theta$, what's the chance I'll experience trajectory $\tau$?"
- **As likelihood**: "Given I observed trajectory $\tau$, how well does policy $\pi_\theta$ explain it?"

---

## 2. Decomposing the Trajectory Likelihood

A trajectory is a complete sequence:

$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T)$$

The full likelihood decomposes as:

$$P(\tau; \theta) = P(s_0) \cdot \prod_{t=0}^{T} \pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t, a_t)$$

**Where:**
- $P(s_0)$ = probability of initial state (determined by environment)
- $\pi_\theta(a_t|s_t)$ = **policy probability** (what we control with $\theta$)
- $P(s_{t+1}|s_t, a_t)$ = **environment dynamics** (transition probability, unknown)
- $\prod$ = product over all time steps

**Three components:**
1. **Initial state distribution**: Where we start (given)
2. **Policy**: Our agent's action choices (learnable)
3. **Environment dynamics**: How the world responds (unknown but doesn't matter!)

---

## 3. The Objective Function

We want to maximize expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim P(\tau; \theta)}[R(\tau)] = \sum_{\tau} P(\tau; \theta) \cdot R(\tau)$$

**Terms explained:**
- $J(\theta)$ = objective function (performance metric)
- $\mathbb{E}[\cdot]$ = expectation (weighted average over all possible trajectories)
- $R(\tau)$ = return of trajectory $\tau$ = $\sum_{t=0}^{T} r_t$ (total reward)
- $P(\tau; \theta)$ = likelihood of trajectory $\tau$

**In plain English**: "What's the average total reward we expect when following policy $\pi_\theta$?"

---

## 4. The Policy Gradient Derivation

### Step 1: Gradient of the Objective

We want to find:

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} P(\tau; \theta) \cdot R(\tau)$$

Since $R(\tau)$ doesn't depend on $\theta$:

$$\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta P(\tau; \theta) \cdot R(\tau)$$

### Step 2: The Log-Derivative Trick

This is a **key likelihood technique** from statistics. Using the identity:

$$\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \cdot \nabla_\theta \log P(\tau; \theta)$$

**Why this works** (calculus):
$$\frac{d}{dx}[\log f(x)] = \frac{1}{f(x)} \cdot f'(x) \quad \Rightarrow \quad f'(x) = f(x) \cdot \frac{d}{dx}[\log f(x)]$$

Substituting back:

$$\nabla_\theta J(\theta) = \sum_{\tau} P(\tau; \theta) \cdot \nabla_\theta \log P(\tau; \theta) \cdot R(\tau)$$

This is an expectation over $\tau$:

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim P(\tau; \theta)}[\nabla_\theta \log P(\tau; \theta) \cdot R(\tau)]}$$

### Step 3: Expanding the Log-Likelihood

Take the log of our trajectory likelihood:

$$\log P(\tau; \theta) = \log P(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T} \log P(s_{t+1}|s_t, a_t)$$

Now take the gradient with respect to $\theta$:

$$\nabla_\theta \log P(\tau; \theta) = \nabla_\theta \log P(s_0) + \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T} \nabla_\theta \log P(s_{t+1}|s_t, a_t)$$

### Step 4: The Magic Simplification ✨

**Critical insight**: Only the policy term depends on $\theta$!

- $\nabla_\theta \log P(s_0) = 0$ (initial state doesn't depend on policy parameters)
- $\nabla_\theta \log P(s_{t+1}|s_t, a_t) = 0$ (environment dynamics don't depend on policy parameters)

Therefore:

$$\boxed{\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)}$$

**This is profound!** Even though the full likelihood includes environment dynamics we don't know (and can't differentiate), they **completely vanish** when we take the gradient!

### Step 5: The Policy Gradient Theorem

Combining everything:

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]}$$

This is the **Policy Gradient Theorem**.

---

## 5. Connection to Maximum Likelihood Estimation

### Standard MLE

In supervised learning, Maximum Likelihood Estimation finds parameters that maximize:

$$\theta^* = \arg\max_\theta \sum_{i=1}^{N} \log P(x_i | \theta)$$

**Goal**: Make observed data as probable as possible.

### Policy Gradient as Weighted MLE

Policy Gradient can be viewed as **MLE with importance weighting**:

$$\theta^* = \arg\max_\theta \sum_{\tau} R(\tau) \cdot \log P(\tau; \theta)$$

**Differences from standard MLE:**

| **Aspect** | **Standard MLE** | **Policy Gradient** |
|------------|------------------|---------------------|
| **Goal** | Make observed data probable | Make high-reward trajectories probable |
| **Weighting** | All samples equal | Weighted by return $R(\tau)$ |
| **Data** | Fixed dataset | Generated by current policy |
| **Optimization** | One-shot | Iterative (policy improves) |

**Intuition:**
- **MLE**: "Fit the model to explain observed data"
- **Policy Gradient**: "Fit the policy to prefer high-reward behaviors"

---

## 6. The REINFORCE Algorithm

### Algorithm Overview

REINFORCE (Monte Carlo Policy Gradient) uses this likelihood perspective directly:

**Algorithm:**
1. **Collect trajectory**: Use current policy $\pi_\theta$ to generate episode $\tau = (s_0, a_0, r_0, \ldots, s_T)$
2. **Calculate return**: $R(\tau) = \sum_{t=0}^{T} r_t$
3. **Estimate gradient**:
   $$\hat{g} = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)$$
4. **Update policy**: $\theta \leftarrow \theta + \alpha \cdot \hat{g}$
5. **Repeat**

Where:
- $\hat{g}$ = gradient estimate (sample-based approximation)
- $\alpha$ = learning rate (e.g., 0.001)

### Interpretation as Weighted Log-Likelihood

The REINFORCE update:

$$\theta \leftarrow \theta + \alpha \cdot \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)$$

Is **gradient ascent on weighted log-likelihood**:

#### For a Good Trajectory ($R(\tau) = +100$):
- $\nabla_\theta \log \pi_\theta(a_t|s_t)$ points in direction to **increase** $\pi_\theta(a_t|s_t)$
- Multiply by positive $R(\tau)$: **reinforce these actions**
- Effect: "Make these state-action pairs more likely in the future"

#### For a Bad Trajectory ($R(\tau) = -50$):
- $\nabla_\theta \log \pi_\theta(a_t|s_t)$ points in direction to **increase** $\pi_\theta(a_t|s_t)$
- Multiply by negative $R(\tau)$: **opposite direction**
- Effect: "Make these state-action pairs less likely in the future"

### Visual Representation

```
Good Trajectory τ: R(τ) = +100
┌─────────────────────────────────┐
│ State s₀ → Action a₀            │
│ ∇_θ log π_θ(a₀|s₀) · (+100)    │──→ INCREASE probability
│                                  │
│ State s₁ → Action a₁            │
│ ∇_θ log π_θ(a₁|s₁) · (+100)    │──→ INCREASE probability
│                                  │
│ ... (all actions reinforced)    │
└─────────────────────────────────┘

Bad Trajectory τ': R(τ') = -50
┌─────────────────────────────────┐
│ State s₀ → Action a₀            │
│ ∇_θ log π_θ(a₀|s₀) · (-50)     │──→ DECREASE probability
│                                  │
│ State s₁ → Action a₁            │
│ ∇_θ log π_θ(a₁|s₁) · (-50)     │──→ DECREASE probability
│                                  │
│ ... (all actions discouraged)   │
└─────────────────────────────────┘
```

---

## 7. Why Use Log-Likelihood?

Three critical reasons:

### 1. Numerical Stability
- **Products of probabilities**: $P(\tau) = \pi(a_0|s_0) \cdot \pi(a_1|s_1) \cdot \ldots$ → very small numbers (underflow)
- **Sums of logs**: $\log P(\tau) = \log \pi(a_0|s_0) + \log \pi(a_1|s_1) + \ldots$ → manageable

### 2. Gradient Tractability
The log-derivative trick enables:
$$\nabla P(\tau; \theta) = P(\tau; \theta) \cdot \nabla \log P(\tau; \theta)$$

Without logs, computing $\nabla P(\tau; \theta)$ directly would be intractable!

### 3. Additivity
Products become sums:
$$\log P(\tau; \theta) = \sum_{t} \log \pi_\theta(a_t|s_t)$$

This makes gradients **additive** across time steps, which is computationally efficient.

---

## 8. Variance Reduction: Beyond Basic REINFORCE

### The Problem with High Variance

Basic REINFORCE has high variance because $R(\tau)$ can vary wildly between trajectories, even with the same policy.

### Solution 1: Returns-to-Go

Instead of using full trajectory return $R(\tau)$ for all time steps, use **returns from time $t$ onwards**:

$$G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$$

Where $\gamma \in (0, 1]$ is the discount factor (typically 0.99).

**Update becomes:**
$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

**Intuition**: Actions at time $t$ should only be credited/blamed for **future** rewards, not past ones.

### Solution 2: Baseline Subtraction

Subtract a baseline $b(s_t)$ to reduce variance:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]$$

**Common choice**: $b(s_t) = V(s_t)$ (value function)

This gives us the **advantage function**:
$$A(s_t, a_t) = G_t - V(s_t)$$

**Interpretation**: "How much better is action $a_t$ compared to the average action in state $s_t$?"

- $A > 0$: Action better than average → increase probability
- $A < 0$: Action worse than average → decrease probability
- $A \approx 0$: Action average → minimal update

---

## 9. From REINFORCE to Actor-Critic

### The Actor-Critic Architecture

Instead of using Monte Carlo returns $G_t$, we can use a **learned critic** to estimate values:

**Two components:**
1. **Actor**: Policy $\pi(a|s; \theta)$ (the likelihood model)
2. **Critic**: Value function $V(s; w)$ (baseline estimator)

### TD Error as Advantage Estimate

The **TD (Temporal Difference) error** serves as a low-variance advantage estimate:

$$\delta_t = r_t + \gamma V(s_{t+1}; w) - V(s_t; w)$$

This approximates the advantage:
$$A(s_t, a_t) \approx \delta_t$$

### Actor-Critic Update Rules

**Actor update** (policy gradient with TD error):
$$\theta \leftarrow \theta + \alpha_\theta \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t$$

**Critic update** (minimize value prediction error):
$$w \leftarrow w + \alpha_w \cdot \delta_t \cdot \nabla_w V(s_t; w)$$

Where:
- $\alpha_\theta$ = actor learning rate
- $\alpha_w$ = critic learning rate
- $\delta_t$ = TD error (advantage estimate)

**Connection to likelihood**: The actor update is still weighted log-likelihood gradient ascent, but now weighted by the **advantage** (TD error) instead of full return.

---

## 10. Summary: The Complete Picture

### Policy Gradient as Likelihood Optimization

$$\boxed{
\begin{align}
\text{Objective:} \quad & J(\theta) = \mathbb{E}_{\tau \sim P(\tau; \theta)}[R(\tau)] \\[0.5em]
\text{Gradient:} \quad & \nabla_\theta J(\theta) = \mathbb{E}_{\tau}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right] \\[0.5em]
\text{REINFORCE:} \quad & \theta \leftarrow \theta + \alpha \cdot \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau) \\[0.5em]
\text{With Baseline:} \quad & \theta \leftarrow \theta + \alpha \cdot \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t)) \\[0.5em]
\text{Actor-Critic:} \quad & \theta \leftarrow \theta + \alpha \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t
\end{align}
}$$

### Key Insights

1. **Likelihood perspective**: Policy gradient is weighted MLE where good trajectories get higher weight

2. **Log-derivative trick**: Enables gradient computation without knowing environment dynamics

3. **Environment dynamics vanish**: $\nabla_\theta \log P(s_{t+1}|s_t, a_t) = 0$ — we only need to differentiate the policy!

4. **From REINFORCE to Actor-Critic**: Progressive variance reduction through better advantage estimates

5. **Unified view**: All policy gradient methods are doing weighted log-likelihood gradient ascent

---

## 11. Notation Reference

| **Symbol** | **Meaning** |
|------------|-------------|
| $\pi_\theta(a\|s)$ | Policy: probability of action $a$ in state $s$ with parameters $\theta$ |
| $\theta$ | Policy parameters (neural network weights) |
| $\tau$ | Trajectory: $(s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$ |
| $R(\tau)$ | Return: total reward of trajectory $\tau$ |
| $P(\tau; \theta)$ | Likelihood: probability of trajectory $\tau$ under policy $\pi_\theta$ |
| $J(\theta)$ | Objective: expected return under policy $\pi_\theta$ |
| $\nabla_\theta$ | Gradient with respect to $\theta$ |
| $\mathbb{E}[\cdot]$ | Expectation (weighted average) |
| $G_t$ | Return-to-go: discounted sum of future rewards from time $t$ |
| $\gamma$ | Discount factor (typically 0.99) |
| $V(s)$ | Value function: expected return from state $s$ |
| $A(s, a)$ | Advantage: how much better is action $a$ than average in state $s$ |
| $\delta_t$ | TD error: $r_t + \gamma V(s_{t+1}) - V(s_t)$ |
| $\alpha$ | Learning rate (step size) |

---

## Conclusion

Understanding policy gradients through the lens of **likelihood optimization** reveals their elegant mathematical structure. The key insight is that we're performing **maximum likelihood estimation on trajectories, weighted by their returns**. This perspective unifies REINFORCE, Actor-Critic, and modern methods like PPO under a common framework.

The log-derivative trick and the vanishing of environment dynamics gradients make this computationally tractable, while variance reduction techniques (baselines, critics) make it practically effective.

**You're doing weighted MLE — just with a clever twist! 🎯**
