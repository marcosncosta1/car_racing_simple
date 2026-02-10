"""
DQN for CarRacing-v3 (Discrete)
================================
A simple, from-scratch Deep Q-Network to learn CarRacing.
Based on the DQN approach from Nathan Bailey's Flappy Bird tutorial,
adapted for the CarRacing-v3 environment with discrete actions.

Actions (5 discrete):
  0 = do nothing
  1 = steer left
  2 = steer right
  3 = gas
  4 = brake

Observation: 96x96 RGB image -> preprocessed to 84x84 grayscale
We stack 4 consecutive frames to capture motion (speed/direction).

=== TUNABLE PARAMETERS (all collected at the top) ===
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym
import cv2
import os
import time
import json

# ============================================================
# HYPERPARAMETERS — Tune these!
# ============================================================

# --- Environment ---
FRAME_SKIP = 4            # repeat each action for N frames (speeds up training)
FRAME_STACK = 4           # number of frames stacked as input (captures motion)
IMAGE_SIZE = 84           # resize frames to IMAGE_SIZE x IMAGE_SIZE

# --- Exploration ---
EPSILON_START = 1.0       # initial exploration rate (100% random)
EPSILON_END = 0.05        # minimum exploration rate
EPSILON_DECAY = 100_000    # steps over which epsilon decays linearly

# --- Learning ---
LEARNING_RATE = 1e-4      # Adam optimizer learning rate
GAMMA = 0.99              # discount factor (how much to value future rewards)
BATCH_SIZE = 32           # minibatch size for training
TARGET_UPDATE = 1000      # update target network every N steps

# --- Replay Buffer ---
BUFFER_SIZE = 200_000      # max transitions stored (keep small for fast experiments)
MIN_BUFFER = 5_000        # minimum transitions before training starts

# --- Training ---
MAX_EPISODES = 5000        # total episodes to train
MAX_STEPS = 1000          # max steps per episode (prevents endless episodes)
SAVE_EVERY = 50           # save model checkpoint every N episodes
NEGATIVE_REWARD_PATIENCE = 100  # end episode after N consecutive negative rewards

# --- Device ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_frame(frame):
    """Convert 96x96x3 RGB frame to 84x84 grayscale float."""
    # Crop bottom 12 rows (score bar) -> 84x96x3
    frame = frame[:84, :, :]
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize to IMAGE_SIZE x IMAGE_SIZE
    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    # Normalize to [0, 1]
    return gray.astype(np.float32) / 255.0


class FrameStack:
    """Maintains a stack of the last N preprocessed frames."""

    def __init__(self, n_frames=FRAME_STACK):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

    def reset(self, frame):
        """Fill stack with copies of the initial frame."""
        processed = preprocess_frame(frame)
        for _ in range(self.n_frames):
            self.frames.append(processed)
        return self._get_state()

    def push(self, frame):
        """Add a new frame and return the stacked state."""
        self.frames.append(preprocess_frame(frame))
        return self._get_state()

    def _get_state(self):
        """Return stacked frames as (n_frames, H, W) numpy array."""
        return np.array(self.frames)


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQN NETWORK
# ============================================================

class DQN(nn.Module):
    """
    CNN-based Q-network.

    Architecture (tune layer sizes and count!):
      Conv2D layers extract spatial features from stacked frames.
      Fully connected layers map features to Q-values for each action.

    Input shape:  (batch, FRAME_STACK, IMAGE_SIZE, IMAGE_SIZE)
    Output shape: (batch, 5)  — one Q-value per discrete action
    """

    def __init__(self, n_actions=5):
        super().__init__()

        # --- CNN feature extractor ---
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),  # 84 -> 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # 20 -> 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # 9 -> 7
            nn.ReLU(),
        )

        # Calculate flattened size after convolutions
        self._conv_out_size = self._get_conv_out_size()

        # --- Fully connected head ---
        self.fc = nn.Sequential(
            nn.Linear(self._conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_out_size(self):
        """Compute output size of conv layers by passing a dummy tensor."""
        dummy = torch.zeros(1, FRAME_STACK, IMAGE_SIZE, IMAGE_SIZE)
        return self.conv(dummy).view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# AGENT
# ============================================================

class DQNAgent:
    """DQN Agent with epsilon-greedy exploration and target network."""

    def __init__(self, n_actions=5):
        self.n_actions = n_actions
        self.step_count = 0

        # Q-network and target network
        self.q_net = DQN(n_actions).to(DEVICE)
        self.target_net = DQN(n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)

        # Replay buffer
        self.buffer = ReplayBuffer()

    def get_epsilon(self):
        """Linear epsilon decay."""
        return max(EPSILON_END,
                   EPSILON_START - self.step_count * (EPSILON_START - EPSILON_END) / EPSILON_DECAY)

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        eps = self.get_epsilon()
        if random.random() < eps:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.q_net(state_t)
                return q_values.argmax(dim=1).item()

    def train_step(self):
        """Sample a batch and perform one gradient step."""
        if len(self.buffer) < MIN_BUFFER:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample()

        states_t = torch.FloatTensor(states).to(DEVICE)
        actions_t = torch.LongTensor(actions).to(DEVICE)
        rewards_t = torch.FloatTensor(rewards).to(DEVICE)
        next_states_t = torch.FloatTensor(next_states).to(DEVICE)
        dones_t = torch.FloatTensor(dones).to(DEVICE)

        # Current Q values: Q(s, a)
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            best_actions = self.q_net(next_states_t).argmax(dim=1) # best action from Q-network Double DQN
            next_q = self.target_net(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze(1) # best action from target network Double DQN

            target = rewards_t + GAMMA * next_q * (1 - dones_t)

        # Loss: MSE between predicted and target Q-values
        loss = nn.MSELoss()(q_values, target)

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (stabilizes training)
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network periodically
        self.step_count += 1
        if self.step_count % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint['step_count']


# ============================================================
# TRAINING LOOP
# ============================================================

def train():
    """Main training loop."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode=None)
    agent = DQNAgent(n_actions=5)
    frame_stack = FrameStack()

    os.makedirs("checkpoints", exist_ok=True)

    # Tracking
    all_rewards = []
    best_avg = -float("inf")

    # Metrics log — saved to metrics.json after training
    metrics = {
        "config": {
            "algorithm": "DQN",
            "frame_skip": FRAME_SKIP,
            "frame_stack": FRAME_STACK,
            "image_size": IMAGE_SIZE,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "epsilon_decay": EPSILON_DECAY,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "batch_size": BATCH_SIZE,
            "buffer_size": BUFFER_SIZE,
            "target_update": TARGET_UPDATE,
            "max_episodes": MAX_EPISODES,
            "max_steps": MAX_STEPS,
            "negative_reward_patience": NEGATIVE_REWARD_PATIENCE,
            "device": str(DEVICE),
        },
        "episodes": [],
    }

    print("\n" + "=" * 60)
    print("TRAINING CONFIG")
    print("=" * 60)
    print(f"  FRAME_SKIP:    {FRAME_SKIP}")
    print(f"  FRAME_STACK:   {FRAME_STACK}")
    print(f"  IMAGE_SIZE:    {IMAGE_SIZE}")
    print(f"  EPSILON:       {EPSILON_START} -> {EPSILON_END} over {EPSILON_DECAY} steps")
    print(f"  LEARNING_RATE: {LEARNING_RATE}")
    print(f"  GAMMA:         {GAMMA}")
    print(f"  BATCH_SIZE:    {BATCH_SIZE}")
    print(f"  BUFFER_SIZE:   {BUFFER_SIZE}")
    print(f"  TARGET_UPDATE: {TARGET_UPDATE}")
    print(f"  MAX_EPISODES:  {MAX_EPISODES}")
    print(f"  DEVICE:        {DEVICE}")
    print("=" * 60 + "\n")

    start_time = time.time()

    for episode in range(1, MAX_EPISODES + 1):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        episode_reward = 0
        neg_reward_count = 0
        episode_losses = []
        action_counts = [0] * 5  # track action distribution

        for step in range(MAX_STEPS):
            # Select and execute action (with frame skipping)
            action = agent.select_action(state)
            action_counts[action] += 1
            total_reward = 0
            done = False

            for _ in range(FRAME_SKIP):
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                if done:
                    break

            next_state = frame_stack.push(obs)

            # Store transition
            agent.buffer.push(state, action, total_reward, next_state, done)

            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            episode_reward += total_reward

            # Early termination: too many negative rewards in a row
            if total_reward < 0:
                neg_reward_count += 1
            else:
                neg_reward_count = 0
            if neg_reward_count >= NEGATIVE_REWARD_PATIENCE:
                break

            if done:
                break

        all_rewards.append(episode_reward)
        avg_reward = np.mean(all_rewards[-20:])
        avg_100 = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
        eps = agent.get_epsilon()
        elapsed = time.time() - start_time
        ep_steps = step + 1
        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

        # Log episode metrics
        metrics["episodes"].append({
            "episode": episode,
            "reward": float(episode_reward),
            "avg_reward_20": float(avg_reward),
            "avg_reward_100": float(avg_100),
            "epsilon": float(eps),
            "loss": avg_loss,
            "steps": ep_steps,
            "total_steps": agent.step_count,
            "buffer_size": len(agent.buffer),
            "elapsed_seconds": float(elapsed),
            "action_distribution": action_counts,
        })

        print(f"Episode {episode:4d} | "
              f"Reward: {episode_reward:7.1f} | "
              f"Avg(20): {avg_reward:7.1f} | "
              f"Eps: {eps:.3f} | "
              f"Buffer: {len(agent.buffer):6d} | "
              f"Steps: {agent.step_count:7d} | "
              f"Time: {elapsed:6.0f}s")

        # Save best model
        if avg_reward > best_avg and episode >= 20:
            best_avg = avg_reward
            agent.save("checkpoints/best_model.pth")
            print(f"  >>> New best average reward: {best_avg:.1f}")

        # Periodic save
        if episode % SAVE_EVERY == 0:
            agent.save(f"checkpoints/model_ep{episode}.pth")

        # Save metrics periodically (every 10 episodes) so data is available during training
        if episode % 10 == 0:
            with open("metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

    # Final saves
    agent.save("checkpoints/final_model.pth")
    metrics["summary"] = {
        "best_avg_reward_20": float(best_avg),
        "final_reward": float(all_rewards[-1]),
        "final_avg_reward_100": float(avg_100),
        "total_training_steps": agent.step_count,
        "total_training_time_seconds": float(time.time() - start_time),
        "max_reward": float(max(all_rewards)),
        "min_reward": float(min(all_rewards)),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    env.close()

    print("\n" + "=" * 60)
    print(f"Training complete! Best avg(20) reward: {best_avg:.1f}")
    print(f"Metrics saved to metrics.json")
    print("=" * 60)

    return all_rewards


# ============================================================
# EVALUATION (watch trained agent play)
# ============================================================

def evaluate(model_path="checkpoints/best_model.pth", episodes=5):
    """Load a trained model and watch it play."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode="human")
    agent = DQNAgent(n_actions=5)
    agent.load(model_path)
    frame_stack = FrameStack()

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        total_reward = 0

        for step in range(MAX_STEPS):
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                action = agent.q_net(state_t).argmax(dim=1).item()

            for _ in range(FRAME_SKIP):
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            state = frame_stack.push(obs)

            if terminated or truncated:
                break

        print(f"Eval Episode {ep}: Reward = {total_reward:.1f}")

    env.close()


# ============================================================
# RECORD (save .mp4 videos of trained agent)
# ============================================================

def record(model_path="checkpoints/best_model.pth", episodes=3):
    """Load a trained model and record gameplay as .mp4 videos."""
    os.makedirs("recordings", exist_ok=True)

    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="recordings",
        name_prefix="car_racing",
        episode_trigger=lambda ep: True,  # record every episode
    )

    agent = DQNAgent(n_actions=5)
    agent.load(model_path)
    frame_stack = FrameStack()

    print(f"\nRecording {episodes} episodes using model: {model_path}")
    print(f"Videos will be saved to recordings/\n")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        total_reward = 0

        for step in range(MAX_STEPS):
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                action = agent.q_net(state_t).argmax(dim=1).item()

            for _ in range(FRAME_SKIP):
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            state = frame_stack.push(obs)

            if terminated or truncated:
                break

        print(f"Recorded Episode {ep}: Reward = {total_reward:.1f}")

    env.close()
    print(f"\nDone! Videos saved in recordings/")


# ============================================================
# PLOT (generate comparison-ready charts from metrics.json)
# ============================================================

def plot(metrics_path="metrics.json"):
    """Generate training charts from saved metrics for hackathon comparison."""
    import matplotlib.pyplot as plt

    with open(metrics_path) as f:
        metrics = json.load(f)

    episodes_data = metrics["episodes"]
    config = metrics["config"]
    summary = metrics.get("summary", {})

    eps = [d["episode"] for d in episodes_data]
    rewards = [d["reward"] for d in episodes_data]
    avg20 = [d["avg_reward_20"] for d in episodes_data]
    avg100 = [d["avg_reward_100"] for d in episodes_data]
    losses = [d["loss"] for d in episodes_data]
    epsilons = [d["epsilon"] for d in episodes_data]
    steps = [d["steps"] for d in episodes_data]
    action_dists = [d["action_distribution"] for d in episodes_data]

    os.makedirs("plots", exist_ok=True)

    # --- Config text box (shown on reward curve) ---
    config_text = (
        f"LR={config['learning_rate']}  \u03b3={config['gamma']}  "
        f"Batch={config['batch_size']}\n"
        f"Buffer={config['buffer_size']}  "
        f"Target update={config['target_update']}\n"
        f"\u03b5: {config['epsilon_start']}\u2192{config['epsilon_end']} "
        f"over {config['epsilon_decay']} steps\n"
        f"Frame skip={config['frame_skip']}  "
        f"Frame stack={config['frame_stack']}  "
        f"Img={config['image_size']}"
    )

    # --- Figure 1: Reward curve (the main comparison chart) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eps, rewards, alpha=0.3, color="steelblue", label="Episode reward")
    ax.plot(eps, avg20, color="darkorange", linewidth=2, label="Avg (20 episodes)")
    ax.plot(eps, avg100, color="crimson", linewidth=2, label="Avg (100 episodes)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    if summary:
        ax.axhline(y=summary.get("best_avg_reward_20", 0), color="green",
                    linestyle=":", alpha=0.7, label=f"Best avg(20): {summary.get('best_avg_reward_20', 0):.1f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("DQN CarRacing-v3 — Reward vs Episode")
    ax.legend(loc="lower right")
    ax.text(0.02, 0.98, config_text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/reward_curve.png", dpi=150)
    print("Saved plots/reward_curve.png")

    # --- Figure 2: Training dashboard (4 subplots) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 2a: Loss curve
    ax = axes[0, 0]
    nonzero_losses = [(e, l) for e, l in zip(eps, losses) if l > 0]
    if nonzero_losses:
        le, lv = zip(*nonzero_losses)
        ax.plot(le, lv, alpha=0.4, color="steelblue")
        # smoothed loss
        window = min(20, len(lv))
        if window > 1:
            smooth = np.convolve(lv, np.ones(window)/window, mode="valid")
            ax.plot(list(le)[window-1:], smooth, color="crimson", linewidth=2, label=f"Smoothed ({window})")
            ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # 2b: Epsilon decay
    ax = axes[0, 1]
    ax.plot(eps, epsilons, color="green", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate (Epsilon)")
    ax.grid(True, alpha=0.3)

    # 2c: Episode length
    ax = axes[1, 0]
    ax.plot(eps, steps, alpha=0.4, color="steelblue")
    window = min(20, len(steps))
    if window > 1:
        smooth_steps = np.convolve(steps, np.ones(window)/window, mode="valid")
        ax.plot(eps[window-1:], smooth_steps, color="darkorange", linewidth=2, label=f"Smoothed ({window})")
        ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length")
    ax.grid(True, alpha=0.3)

    # 2d: Action distribution (stacked area)
    ax = axes[1, 1]
    action_labels = ["Nothing", "Left", "Right", "Gas", "Brake"]
    action_arrays = np.array(action_dists, dtype=float)
    # Normalize to percentages per episode
    row_sums = action_arrays.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    action_pcts = action_arrays / row_sums * 100
    ax.stackplot(eps, action_pcts.T, labels=action_labels,
                 colors=["gray", "royalblue", "coral", "limegreen", "orange"], alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Action %")
    ax.set_title("Action Distribution")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"DQN Training Dashboard — {config['max_episodes']} episodes\n{config_text}",
                 fontsize=12, y=1.04, fontfamily="monospace")
    fig.tight_layout()
    fig.savefig("plots/training_dashboard.png", dpi=150)
    print("Saved plots/training_dashboard.png")

    # --- Print summary table ---
    if summary:
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"  Best avg(20) reward:  {summary.get('best_avg_reward_20', 'N/A'):.1f}")
        print(f"  Final avg(100) reward:{summary.get('final_avg_reward_100', 'N/A'):>7.1f}")
        print(f"  Max episode reward:   {summary.get('max_reward', 'N/A'):.1f}")
        print(f"  Min episode reward:   {summary.get('min_reward', 'N/A'):.1f}")
        print(f"  Total training steps: {summary.get('total_training_steps', 'N/A')}")
        total_secs = summary.get('total_training_time_seconds', 0)
        print(f"  Total training time:  {total_secs/60:.1f} min")
        print("=" * 50)

    plt.close("all")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        model = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/best_model.pth"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        evaluate(model, n)
    elif len(sys.argv) > 1 and sys.argv[1] == "record":
        model = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/best_model.pth"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        record(model, n)
    elif len(sys.argv) > 1 and sys.argv[1] == "plot":
        path = sys.argv[2] if len(sys.argv) > 2 else "metrics.json"
        plot(path)
    else:
        train()
