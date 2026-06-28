"""
PPO training with VisionEncoder (perception) + MLP policy (navigation).

Two modes:
    Frozen encoder (default --encoder-lr 0):
        PixelGameEnv → [CNN, frozen] → 64-dim → [MLP, trains] → action
    Fine-tune encoder (--encoder-lr 2.5e-5):
        PixelGameEnv → [CNN, trains slowly] → 64-dim → [MLP, trains] → action
        Raw frames are stored in the rollout buffer and re-encoded with gradients
        during the PPO update so both networks learn end-to-end.

Two reward signals are logged:
    ep_rew   — noisy per-episode reward from random-seed training envs
    eval_rew — clean mean ± std over N fixed-seed episodes (every --eval-interval iters)

Usage:
    python scripts/train_ppo_vision.py
    python scripts/train_ppo_vision.py --encoder-lr 2.5e-5 --save-dir models/ppo_vision_run4 --log-file logs/train_vision_run4.log
    python scripts/train_ppo_vision.py --eval-interval 20 --n-eval 16
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv

from envs.pixel_env import PixelGameEnv
from models.vision_encoder import VisionEncoder
from models.mlp_policy import MLPActorCritic
from models.ppo_mlp import VectorPPO
from models.ppo import PPOConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _log(msg: str, log_file: str | None) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


@torch.no_grad()
def encode_obs(frames_np: np.ndarray, encoder: VisionEncoder, device: torch.device) -> torch.Tensor:
    """(N, 84, 84, 3) uint8  →  (N, 64) float32 on device."""
    t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().div(255.0).to(device)
    return encoder(t)


def save_checkpoint(policy, optimizer, embed_dim, n_actions, iteration,
                    global_step, best_mean_reward, save_dir, tag="", encoder=None):
    os.makedirs(save_dir, exist_ok=True)
    name = f"checkpoint_{tag}.pt" if tag else "checkpoint_latest.pt"
    path = os.path.join(save_dir, name)
    data = {
        "policy_state_dict":    policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "embed_dim":            embed_dim,
        "n_actions":            n_actions,
        "iteration":            iteration,
        "global_step":          global_step,
        "best_mean_reward":     best_mean_reward,
    }
    if encoder is not None:
        data["encoder_state_dict"] = encoder.state_dict()
    torch.save(data, path)
    return path


@torch.no_grad()
def run_eval(
    encoder: VisionEncoder,
    policy: MLPActorCritic,
    eval_envs,
    eval_seeds: list[int],
    device: torch.device,
    max_steps: int = 1000,
) -> np.ndarray:
    """Run one episode per fixed seed, return array of episode rewards."""
    n = len(eval_seeds)
    obs_raw, _ = eval_envs.reset(seed=eval_seeds)
    obs_enc = encode_obs(obs_raw, encoder, device)

    ep_rewards = np.zeros(n)
    ep_done    = np.zeros(n, dtype=bool)
    final_rew  = np.full(n, np.nan)

    for _ in range(max_steps):
        action, _, _, _ = policy.get_action_and_value(obs_enc)
        obs_raw, rewards, terminated, truncated, _ = eval_envs.step(action.cpu().numpy())

        for i in range(n):
            if not ep_done[i]:
                ep_rewards[i] += rewards[i]
                if terminated[i] or truncated[i]:
                    final_rew[i] = ep_rewards[i]
                    ep_done[i]   = True

        if ep_done.all():
            break

        obs_enc = encode_obs(obs_raw, encoder, device)

    # Episodes that hit max_steps without terminating
    final_rew = np.where(np.isnan(final_rew), ep_rewards, final_rew)
    return final_rew


def finetune_update(
    encoder: VisionEncoder,
    policy: MLPActorCritic,
    optimizer: torch.optim.Optimizer,
    raw_frames: np.ndarray,       # (T, M, H, W, C) uint8
    buffer,                        # VectorRolloutBuffer (GAE already computed)
    config,
    device: torch.device,
) -> dict[str, float]:
    """
    PPO update that flows gradients through the encoder.

    Unlike VectorPPO.update() (which feeds pre-encoded 64-dim obs), this
    function re-encodes raw frames inside each minibatch so the backward pass
    reaches the CNN weights.
    """
    T, M = buffer.n_steps, buffer.n_envs
    total = T * M

    raw_flat = raw_frames.reshape(total, *raw_frames.shape[2:])   # (total, H, W, C)
    actions_flat    = buffer.actions.view(total)
    log_probs_flat  = buffer.log_probs.view(total)
    returns_flat    = buffer.returns.view(total)
    values_flat     = buffer.values.view(total)

    adv = buffer.advantages.view(total)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    metrics: dict[str, list[float]] = {
        k: [] for k in ["policy_loss", "value_loss", "entropy_loss", "approx_kl"]
    }

    for _ in range(config.n_epochs):
        indices = torch.randperm(total, device=device)
        for start in range(0, total, config.minibatch_size):
            idx = indices[start : start + config.minibatch_size]

            # Re-encode raw frames WITH gradients so encoder gets updated
            mb_raw = raw_flat[idx.cpu().numpy()]                       # numpy (B, H, W, C)
            mb_t   = torch.from_numpy(mb_raw).permute(0, 3, 1, 2).float().div(255.0).to(device)
            mb_obs = encoder(mb_t)                                     # (B, 64) — grad flows here

            _, new_log_prob, entropy, new_value = policy.get_action_and_value(
                mb_obs, actions_flat[idx]
            )
            new_value = new_value.squeeze(-1)

            log_ratio  = new_log_prob - log_probs_flat[idx]
            ratio      = log_ratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()

            mb_adv = adv[idx]
            surr1  = ratio * mb_adv
            surr2  = torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            v_clip      = values_flat[idx] + torch.clamp(
                new_value - values_flat[idx], -config.clip_coef, config.clip_coef
            )
            value_loss  = 0.5 * torch.max(
                (new_value - returns_flat[idx]) ** 2,
                (v_clip    - returns_flat[idx]) ** 2,
            ).mean()

            entropy_loss = -entropy.mean()
            loss = (
                policy_loss
                + config.value_coef   * value_loss
                + config.entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(encoder.parameters()),
                config.max_grad_norm,
            )
            optimizer.step()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy_loss"].append(entropy_loss.item())
            metrics["approx_kl"].append(approx_kl)

    with torch.no_grad():
        ev = (1.0 - (returns_flat - values_flat).var() / (returns_flat.var() + 1e-8)).item()

    return {k: float(np.mean(v)) for k, v in metrics.items()} | {"explained_variance": ev}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        # Write header
        with open(args.log_file, "w") as f:
            f.write(f"  {'iter':>5} | {'steps':>10} | {'sps':>6} | {'ep_rew':>7} | "
                    f"{'ep_len':>6} | {'n_eps':>5} | {'pol':>8} | {'val':>8} | "
                    f"{'ent':>8} | {'kl':>10} | {'ev':>6} | {'lr':>10}\n")

    # ── Encoder ──────────────────────────────────────────────────────────────
    encoder = VisionEncoder.load(args.encoder, device=device)
    encoder.to(device)
    embed_dim = encoder.embed_dim
    finetune = args.encoder_lr > 0
    if finetune:
        encoder.train()
        _log(f"Encoder loaded from {args.encoder}  (embed_dim={embed_dim}, fine-tune lr={args.encoder_lr:.1e})", args.log_file)
    else:
        encoder.eval()
        encoder.freeze()
        _log(f"Encoder loaded from {args.encoder}  (embed_dim={embed_dim}, frozen)", args.log_file)

    # ── Environments ─────────────────────────────────────────────────────────
    envs = SyncVectorEnv([
        lambda: PixelGameEnv(frame_size=(84, 84), render_mode="rgb_array", max_steps=1000)
        for _ in range(args.num_envs)
    ])
    n_actions = envs.single_action_space.n

    # ── Eval environments (fixed seeds, never used for training) ─────────────
    eval_seeds = list(range(1000, 1000 + args.n_eval))
    eval_envs  = SyncVectorEnv([
        lambda: PixelGameEnv(frame_size=(84, 84), render_mode="rgb_array", max_steps=1000)
        for _ in range(args.n_eval)
    ])

    # ── Policy + PPO ─────────────────────────────────────────────────────────
    policy = MLPActorCritic(
        obs_dim   = embed_dim,
        n_actions = n_actions,
        hidden    = (256, 128),
    ).to(device)

    config = PPOConfig(
        n_steps       = args.n_steps,
        n_envs        = args.num_envs,
        n_epochs      = 4,
        minibatch_size= 256,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_coef     = 0.2,
        value_coef    = 0.5,
        entropy_coef  = args.entropy_coef,
        max_grad_norm = 0.5,
        learning_rate = args.lr,
        anneal_lr     = not args.no_anneal_lr,
        clip_value_loss = True,
    )

    ppo = VectorPPO(policy, config, obs_shape=(embed_dim,), device=device)

    if finetune:
        # Replace single-group optimizer with two-group version.
        # initial_lr is stored per-group so anneal_lr scales each correctly.
        from torch.optim import Adam as _Adam
        ppo.optimizer = _Adam([
            {"params": policy.parameters(),  "lr": args.lr,         "initial_lr": args.lr},
            {"params": encoder.parameters(), "lr": args.encoder_lr, "initial_lr": args.encoder_lr},
        ], eps=1e-5)

    steps_per_iter = args.n_steps * args.num_envs
    total_iters    = args.total_timesteps // steps_per_iter

    _log(f"Device        : {device}", args.log_file)
    _log(f"Environments  : {args.num_envs}", args.log_file)
    _log(f"Steps/iter    : {steps_per_iter:,}", args.log_file)
    _log(f"Total iters   : {total_iters:,}", args.log_file)
    _log(f"Total steps   : {total_iters * steps_per_iter:,}", args.log_file)
    _log(f"Eval envs     : {args.n_eval}  (seeds {eval_seeds[0]}–{eval_seeds[-1]}, every {args.eval_interval} iters)", args.log_file)
    _log(f"Encoder mode  : {'fine-tune (lr=' + str(args.encoder_lr) + ')' if finetune else 'frozen'}", args.log_file)

    # ── Initial obs ──────────────────────────────────────────────────────────
    raw_obs, _ = envs.reset(seed=42)
    obs_enc    = encode_obs(raw_obs, encoder, device)   # (n_envs, 64)
    done       = torch.zeros(args.num_envs, device=device)

    # Episode tracking
    ep_rewards = np.zeros(args.num_envs)
    ep_lengths = np.zeros(args.num_envs, dtype=int)

    best_mean_reward = -np.inf
    best_eval_reward = -np.inf
    global_step      = 0
    t_start          = time.time()

    # Raw frame buffer for fine-tuning (stores uint8 pixels for re-encoding with grad)
    if finetune:
        raw_frames_buf = np.zeros(
            (args.n_steps, args.num_envs, 84, 84, 3), dtype=np.uint8
        )

    # ── Main loop ────────────────────────────────────────────────────────────
    for iteration in range(1, total_iters + 1):
        ppo.anneal_lr(iteration - 1, total_iters)
        ppo.buffer.reset()

        completed_rewards: list[float] = []
        completed_lengths: list[int]   = []

        # Rollout collection
        with torch.no_grad():
            for step in range(args.n_steps):
                action, log_prob, _ent, value = policy.get_action_and_value(obs_enc)

                raw_obs, rewards, terminated, truncated, _ = envs.step(action.cpu().numpy())
                done_np = np.logical_or(terminated, truncated).astype(np.float32)

                ep_rewards += rewards
                ep_lengths += 1

                for i, d in enumerate(done_np):
                    if d:
                        completed_rewards.append(float(ep_rewards[i]))
                        completed_lengths.append(int(ep_lengths[i]))
                        ep_rewards[i] = 0
                        ep_lengths[i] = 0

                if finetune:
                    raw_frames_buf[step] = raw_obs   # store uint8 for re-encoding

                next_obs_enc = encode_obs(raw_obs, encoder, device)
                reward_t     = torch.tensor(rewards,  dtype=torch.float32, device=device)
                done_t       = torch.tensor(done_np,  dtype=torch.float32, device=device)

                ppo.buffer.add(obs_enc, action, log_prob, reward_t, done_t, value)
                obs_enc = next_obs_enc
                done    = done_t

        global_step += steps_per_iter

        last_value = policy.get_value(obs_enc)
        ppo.buffer.compute_gae(last_value, done, config.gamma, config.gae_lambda)

        if finetune:
            metrics = finetune_update(
                encoder, policy, ppo.optimizer, raw_frames_buf, ppo.buffer, config, device
            )
        else:
            metrics = ppo.update()

        # ── Logging ──────────────────────────────────────────────────────────
        elapsed = time.time() - t_start
        sps     = int(global_step / elapsed)
        lr_now  = ppo.optimizer.param_groups[0]["lr"]

        ep_rew_str = f"{np.mean(completed_rewards):>7.2f}" if completed_rewards else "    nan"
        ep_len_str = f"{int(np.mean(completed_lengths)):>6}" if completed_lengths else "   nan"
        n_eps      = len(completed_rewards)

        msg = (
            f"iter {iteration:>5} | steps {global_step:>10,} | sps {sps:>6} | "
            f"ep_rew {ep_rew_str} | ep_len {ep_len_str} | n_eps {n_eps:>5} | "
            f"pol {metrics['policy_loss']:>8.4f} | val {metrics['value_loss']:>8.4f} | "
            f"ent {metrics['entropy_loss']:>8.4f} | kl {metrics['approx_kl']:>10.6f} | "
            f"ev {metrics['explained_variance']:>6.3f} | lr {lr_now:.2e}"
        )
        _log(msg, args.log_file)

        # ETA every 50 iters
        if iteration % 50 == 0:
            remaining = (total_iters - iteration) * (elapsed / iteration)
            eta = str(datetime.timedelta(seconds=int(remaining)))
            _log(f"  eta {eta}", args.log_file)

        # Save best (by noisy training ep_rew)
        if completed_rewards:
            mean_rew = float(np.mean(completed_rewards))
            if mean_rew > best_mean_reward:
                best_mean_reward = mean_rew
                path = save_checkpoint(policy, ppo.optimizer, embed_dim, n_actions,
                                       iteration, global_step, best_mean_reward,
                                       args.save_dir, tag="best",
                                       encoder=encoder if finetune else None)
                _log(f"  [best] new best ep_rew {best_mean_reward:.3f} → {path}", args.log_file)

        # Fixed-seed evaluation
        if iteration % args.eval_interval == 0:
            policy.eval()
            encoder.eval()
            eval_rewards = run_eval(encoder, policy, eval_envs, eval_seeds, device)
            policy.train()
            if finetune:
                encoder.train()
            eval_mean = float(np.mean(eval_rewards))
            eval_std  = float(np.std(eval_rewards))
            _log(
                f"  [eval] mean {eval_mean:>8.2f} | std {eval_std:>7.2f} | "
                f"min {float(np.min(eval_rewards)):>8.2f} | max {float(np.max(eval_rewards)):>8.2f}",
                args.log_file,
            )
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                path = save_checkpoint(policy, ppo.optimizer, embed_dim, n_actions,
                                       iteration, global_step, best_eval_reward,
                                       args.save_dir, tag="best_eval",
                                       encoder=encoder if finetune else None)
                _log(f"  [eval] new best eval {best_eval_reward:.3f} → {path}", args.log_file)

    # ── Done ─────────────────────────────────────────────────────────────────
    save_checkpoint(policy, ppo.optimizer, embed_dim, n_actions,
                    total_iters, global_step, best_mean_reward,
                    args.save_dir, tag="final",
                    encoder=encoder if finetune else None)
    _log(f"\nTraining complete — {global_step:,} steps in {time.time()-t_start:.1f}s", args.log_file)
    _log(f"Final checkpoint : {os.path.join(args.save_dir, 'checkpoint_final.pt')}", args.log_file)

    envs.close()
    eval_envs.close()
    import os as _os; _os._exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO with frozen vision encoder")
    parser.add_argument("--encoder",           default="models/vision_encoder.pt")
    parser.add_argument("--total-timesteps",   type=int,   default=3_000_000)
    parser.add_argument("--num-envs",          type=int,   default=8)
    parser.add_argument("--n-steps",           type=int,   default=128)
    parser.add_argument("--lr",                type=float, default=2.5e-4)
    parser.add_argument("--encoder-lr",        type=float, default=0.0,
                        help="LR for fine-tuning the encoder (0 = frozen, default). Try 2.5e-5.")
    parser.add_argument("--entropy-coef",      type=float, default=0.05)
    parser.add_argument("--no-anneal-lr",      action="store_true")
    parser.add_argument("--save-dir",          default="models/ppo_vision")
    parser.add_argument("--log-file",          default="logs/train_vision_run1.log")
    parser.add_argument("--eval-interval",     type=int,   default=20,
                        help="Run fixed-seed eval every N iters (default: 20)")
    parser.add_argument("--n-eval",            type=int,   default=8,
                        help="Number of fixed-seed eval episodes per eval run (default: 8)")
    args = parser.parse_args()
    train(args)
