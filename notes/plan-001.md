# Plan: 2D Game-Playing AI Agent (Pixel-Based)

## Goal
Build a flexible, well-understood reinforcement learning agent that plays custom 2D games using only pixel input (no game state access).

## Tech Stack

- **Language**: Python
- **Framework**: PyTorch (from scratch, not SB3)
- **Reference**: CleanRL (single-file, transparent implementations)
- **Env API**: Gymnasium
- **Algorithm**: PPO (Proximal Policy Optimization)

## Why Not Stable Baselines3

SB3 hides implementation details. The goal is deep understanding + flexibility, not fastest path to a working demo.

## Architecture Components

### 1. Environment Wrapper
- Frame preprocessing (resize, grayscale, normalize to [0,1])
- Frame stacking (4 consecutive frames → captures motion/temporal info)
- Reward clipping (stabilizes training, e.g. clip to [-1, 1])
- Episode/life management

### 2. Neural Network
- CNN backbone for spatial feature extraction from raw pixels
- Actor head → outputs action probability distribution π(a|s)
- Critic head → outputs state value estimate V(s)
- Shared convolutional layers, separate FC heads

### 3. Experience Pipeline
- On-policy rollout buffer (collect full trajectories)
- GAE (Generalized Advantage Estimation) for advantage computation
- Why on-policy: PPO needs fresh data from current policy, no replay buffer like DQN

### 4. Algorithm Core (PPO)
- Collect rollouts from vectorized environments
- Compute advantages via GAE(λ)
- Compute returns
- Update policy for multiple epochs per rollout batch
- PPO clip mechanism: prevents destructive large policy updates
- Why PPO over DQN:
  - Clipping mechanism is intuitive
  - On-policy data is simpler conceptually
  - Works for both discrete and continuous actions
  - Better sample efficiency in practice
  - More transferable knowledge (policy gradients are fundamental)

### 5. Training Loop
- Collect N steps across M parallel envs
- Compute advantages and returns
- Run K PPO epochs over the batch
- Log: loss, entropy, explained variance, mean reward
- Periodic evaluation and checkpointing

## Implementation Phases

| Week | Focus | Deliverables |
|------|-------|-------------|
| 1 | Environment system | Custom env, Gymnasium wrappers (frame stack, preprocessing, reward clip) |
| 2 | PPO from scratch | Core algorithm, CleanRL as reference, training loop |
| 3 | Network + preprocessing | CNN architecture, frame stacking integration, normalization |
| 4 | Training infrastructure | Logging (TensorBoard/W&B), evaluation, checkpointing, video recording |
| 5+ | Extensions | Distributed training, custom CNN experiments, curriculum learning, hyperparameter sweeps |

## Performance Considerations

- **Vectorized environments**: `gymnasium.vector` — 4-16 envs in parallel for 4-16x faster data collection
- **GPU for training**: CNN forward/backward on GPU, env stepping stays on CPU
- **Async pipeline**: Separate process(es) for environment stepping while GPU trains on collected batch

## Key Concepts to Understand Deeply

1. MDP formulation (state, action, reward, transition, discount γ)
2. Policy gradient theorem and why it works
3. Advantage estimation (why not just use raw returns)
4. PPO clipping (why it prevents policy collapse)
5. GAE(λ) — bias-variance tradeoff in advantage estimation
6. Why frame stacking is necessary (Markov property violation from single frame)
7. Why reward clipping helps (prevents gradient explosion from large rewards)
8. Entropy bonus in PPO (encourages exploration)
