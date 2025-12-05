# CARLA PPO Autonomous Driving Agent

End-to-end reinforcement learning stack for teaching an autonomous vehicle to follow lanes, obey speed targets, and avoid collisions in the CARLA simulator. The project wraps CARLA in a Gymnasium-compatible environment, trains a PPO agent (Stable-Baselines3) on RGB imagery, then evaluates, visualizes, and reports the results.

---

## Key Features
- **Rich simulator wrapper**: `CarlaEnv` exposes RGB + semantic cameras, collision sensing, lane offset, and flexible reward shaping.
- **Robust PPO training loop**: automatic checkpointing/resume, per-chunk evaluation, CSV + TensorBoard logging, and dual training scripts (`train_agent.py` for fancy console output).
- **Evaluation & demo tooling**: headless metrics (`evaluate_agent.py`) and interactive demo playback/recording (`demo.py`).
- **Analytics pipeline**: reward plots, final-metric bar charts, and PDF report generation.
- **Pre-trained assets**: `models/` already contains multiple checkpoints; `logs/` ships with progress & TensorBoard files for quick inspection.

---

## Repository Layout

| Path | Purpose |
|------|---------|
| `carla_env.py` | Gymnasium environment that encapsulates CARLA sensors, reward logic, and reset/step lifecycle. |
| `train_agent.py` | Primary training entry point with checkpoint resume, chunked evals, and logging. |
| `evaluate_agent.py` | Batch evaluation over N episodes; writes `logs/eval_metrics.csv`. |
| `demo.py` | Runs a trained agent live, with optional MP4 recording via OpenCV. |
| `plot_training_rewards.py` | Turns `logs/final_metrics.csv` into `final_metrics_graph.png`. |
| `generate_report.py` | Builds `performance_report.pdf` summarizing metrics and methodology. |
| `models/` | Timestamped PPO checkpoints (`ppo_carla_<steps>.zip`), rolling checkpoint (`ppo_carla_checkpoint.zip`), and final model. |
| `logs/` | TensorBoard events, CSV summaries, and sample evaluation outputs. |
| `Video/` | Example demo video (`Demonstration.mp4`). |
| `carla_rl/` | Self-contained copy of the project for packaging/distribution. |
| `test_*.py` | Smoke tests for CARLA connectivity and environment sanity. |

> **Note:** Many assets (images, MP4s, logs) are already generated; feel free to delete them if you need a clean slate.

---

## Prerequisites

- **Python**: 3.9–3.11 (tested with 3.10).
- **CARLA Simulator**: 0.9.8 binary running locally. Launch via `CarlaUE4.exe -quality-level=Epic -carla-port=2000` before any training/evaluation scripts.
- **GPU**: Recommended for PPO (CUDA-capable NVIDIA card). CPU-only works but will be significantly slower.
- **Dependencies** (install via pip):
  ```bash
  pip install --upgrade pip
  pip install carla==0.9.8 gymnasium stable-baselines3 torch torchvision torchaudio \
      tensorboard numpy pandas matplotlib opencv-python reportlab tqdm
  ```
  Adjust versions to match your CARLA build or CUDA toolkit if needed.

---

## Quickstart

1. **Start CARLA server**
   ```bash
   CarlaUE4.exe -quality-level=Epic -carla-port=2000
   ```

2. **Verify connectivity**
   ```bash
   python test_carla_connection.py
   ```
   Expected output: ✅ Connected to CARLA successfully!

3. **Train / resume PPO**
   ```bash
   python train_agent.py --timesteps 300000 --chunk 10000 --eval_eps 3
   ```
   - Saves checkpoints to `models/ppo_carla_<steps>.zip` and `models/ppo_carla_checkpoint.zip`.
   - Logs training progress to `logs/training_rewards.csv` and TensorBoard (`tensorboard --logdir logs`).
   - Auto-resumes from the latest checkpoint if present.

4. **Evaluate a trained model**
   ```bash
   python evaluate_agent.py --model models/ppo_carla_final.zip --episodes 25
   ```
   Metrics are printed and written to `logs/eval_metrics.csv`.

5. **Run an interactive demo (optional recording)**
   ```bash
   python demo.py --model models/ppo_carla_final.zip --record demo_output.mp4
   ```
   Press `q` to exit early. Recording is saved via OpenCV if `--record` is supplied.

6. **Generate visuals & report**
   ```bash
   python plot_training_rewards.py        # builds final_metrics_graph.png
   python generate_report.py              # builds performance_report.pdf
   ```

---

## Logs, Artifacts, and Monitoring
- **TensorBoard**: `tensorboard --logdir logs` to monitor PPO loss, reward trends, collisions, etc.
- **CSV summaries**:
  - `logs/training_rewards.csv`: rolling metrics after each chunk.
  - `logs/final_metrics.csv`: final evaluation from training scripts.
  - `logs/eval_metrics.csv`: standalone evaluation runs.
- **Models**:
  - `models/ppo_carla_<steps>.zip`: timestamped snapshots.
  - `models/ppo_carla_checkpoint.zip`: rolling resume checkpoint.
  - `models/ppo_carla_final.zip`: last training save (used by demo/eval scripts).

---

## Testing & Diagnostics
- `python test_carla_connection.py`: confirms CARLA RPC connectivity.
- `python test_env.py`: basic roll-out with constant throttle (for quick sanity checks; adjust to match the current `CarlaEnv.step` signature if you modify the environment).
- `python test_agent.py`: random-action smoke test (same note about signature).

When making changes to `CarlaEnv`, re-run these quick tests before launching lengthy training jobs.

---

## Troubleshooting Tips
- **Cannot connect to CARLA**: ensure the simulator is running on the same host/port as configured in `CarlaEnv`. Increase `--timeout` in `test_carla_connection.py` or `CarlaEnv._connect_to_carla`.
- **Black camera frames**: allow a few ticks after reset; the environment already advances several frames, but slow machines might need larger `max_episode_steps`.
- **Diverging training**: tune `target_speed_kmh`, reward weights, or episode length in `CarlaEnv`. Reduce `chunk` size for more frequent evaluations.
- **Mismatch in `env.step` signature**: `CarlaEnv` follows Gymnasium’s `(obs, reward, terminated, truncated, info)` return format, while legacy tests expect Gym’s `(obs, reward, done, info)`. Update test scripts accordingly.
- **Performance**: run CARLA and training on separate GPUs or machines when possible; otherwise, lower CARLA graphical settings or use `-quality-level=Low`.

---

## Next Steps / Ideas
- Extend observations with LiDAR or bird’s-eye view maps.
- Integrate traffic participants and traffic-light compliance.
- Expand evaluation suite with waypoint tracking error, comfort metrics, or multi-weather benchmarks.
- Containerize CARLA + trainer for reproducible experiments.

Happy driving! 🚗💡

