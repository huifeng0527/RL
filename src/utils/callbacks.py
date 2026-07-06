"""Training callbacks for reinforcement learning."""

import json
import os
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


class StatusWriter:
    """Append status events and atomically publish the latest snapshot."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "training_status.jsonl"
        self.latest_path = self.run_dir / "training_status_latest.json"
        self.episodes_path = self.run_dir / "recent_episodes_latest.json"
        self.sampled_episodes_path = self.run_dir / "sampled_episodes.jsonl"
        self._episode_buffer = deque(maxlen=300)
        self._episode_count = 0

    def write_event(self, event):
        event = dict(event)
        event.setdefault("timestamp", utc_now_iso())
        try:
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass
        self.write_latest(event)

    def write_episode(self, episode):
        episode = dict(episode)
        episode.setdefault("timestamp", utc_now_iso())
        self._episode_count += 1
        self._episode_buffer.append(episode)

        if self._episode_count % 20 == 0:
            try:
                with self.episodes_path.open("w", encoding="utf-8") as f:
                    json.dump(list(self._episode_buffer), f, ensure_ascii=False)
            except OSError:
                pass

        if self._episode_count % 100 == 0:
            try:
                with self.sampled_episodes_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(episode, ensure_ascii=False) + "\n")
            except OSError:
                pass

    def write_latest(self, status):
        tmp_path = self.latest_path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(status, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.latest_path)
        except OSError:
            pass


class TrainingStatusCallback(BaseCallback):
    """Write dashboard-friendly live training status files."""

    def __init__(
        self,
        run_dir,
        phase,
        iteration,
        total_timesteps,
        log_freq=10000,
        verbose=0,
    ):
        super().__init__(verbose)
        self.writer = StatusWriter(run_dir)
        self.phase = phase
        self.iteration = iteration
        self.total_timesteps = total_timesteps
        self.log_freq = log_freq
        self.recent_episodes = deque(maxlen=200)
        self.done_reasons = Counter()
        self.last_write_time = 0.0

    def _on_training_start(self) -> None:
        self._write_status("phase_started")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        for env_idx, info in enumerate(infos):
            if not info:
                continue
            league_episode = info.get("league_episode")
            if league_episode:
                episode = dict(league_episode)
                episode.update({
                    "event": "episode_completed",
                    "phase": self.phase,
                    "iteration": self.iteration,
                    "env_idx": env_idx,
                    "global_timestep": int(self.num_timesteps),
                })
                self.recent_episodes.append(episode)
                if episode.get("done_reason"):
                    self.done_reasons[str(episode["done_reason"])] += 1
                self.writer.write_episode(episode)

        if self.num_timesteps % self.log_freq == 0:
            self._write_status("phase_progress")
        return True

    def _on_training_end(self) -> None:
        self._write_status("phase_completed")

    def _recent_mean(self, key):
        values = [float(ep[key]) for ep in self.recent_episodes if key in ep and ep[key] is not None]
        return sum(values) / len(values) if values else None

    def _recent_by_opponent(self):
        grouped = {}
        total = max(len(self.recent_episodes), 1)
        numeric_keys = ("episode_length", "selected_opponent_prob", "tis", "zpd_coverage")

        for ep in self.recent_episodes:
            name = ep.get("selected_opponent_name")
            if not name:
                continue
            bucket = grouped.setdefault(name, {"episodes": 0, **{key: [] for key in numeric_keys}})
            bucket["episodes"] += 1
            for key in numeric_keys:
                if ep.get(key) is not None:
                    bucket[key].append(float(ep[key]))

        result = {}
        for name, bucket in grouped.items():
            episodes = bucket["episodes"]
            result[name] = {
                "episodes": episodes,
                "selection_rate": episodes / total,
                "episode_length_mean": sum(bucket["episode_length"]) / len(bucket["episode_length"]) if bucket["episode_length"] else None,
                "selected_prob_mean": sum(bucket["selected_opponent_prob"]) / len(bucket["selected_opponent_prob"]) if bucket["selected_opponent_prob"] else None,
                "tis_mean": sum(bucket["tis"]) / len(bucket["tis"]) if bucket["tis"] else None,
                "zpd_coverage_mean": sum(bucket["zpd_coverage"]) / len(bucket["zpd_coverage"]) if bucket["zpd_coverage"] else None,
            }
        return result

    def _latest_episode(self):
        return dict(self.recent_episodes[-1]) if self.recent_episodes else None

    def _logger_snapshot(self):
        values = getattr(self.model.logger, "name_to_value", {})
        keys = [
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
            "train/loss",
            "train/value_loss",
            "train/policy_gradient_loss",
            "train/entropy_loss",
            "train/approx_kl",
            "auxiliary/prediction_loss",
        ]
        snapshot = {}
        for key in keys:
            if key in values:
                value = values[key]
                try:
                    snapshot[key] = float(value)
                except (TypeError, ValueError):
                    snapshot[key] = value
        return snapshot

    def _write_status(self, event):
        progress = min(float(self.num_timesteps) / max(self.total_timesteps, 1), 1.0)
        latest_episode = self._latest_episode()
        status = {
            "event": event,
            "phase": self.phase,
            "iteration": int(self.iteration),
            "timesteps": int(self.num_timesteps),
            "total_timesteps": int(self.total_timesteps),
            "progress": progress,
            "recent_episode_count": len(self.recent_episodes),
            "recent_tis_mean": self._recent_mean("tis"),
            "recent_zpd_coverage_mean": self._recent_mean("zpd_coverage"),
            "recent_episode_length_mean": self._recent_mean("episode_length"),
            "recent_too_close_rate_mean": self._recent_mean("too_close_rate"),
            "recent_too_far_rate_mean": self._recent_mean("too_far_rate"),
            "recent_by_opponent": self._recent_by_opponent(),
            "done_reason_counts": dict(self.done_reasons),
            "latest_episode": latest_episode,
            "logger": self._logger_snapshot(),
        }
        if latest_episode:
            status["pfsp"] = {
                "selected_opponent_index": latest_episode.get("selected_opponent_index"),
                "selected_opponent_name": latest_episode.get("selected_opponent_name"),
                "selected_opponent_prob": latest_episode.get("selected_opponent_prob"),
                "pfsp_probs": latest_episode.get("pfsp_probs", []),
                "pfsp_pool_size": latest_episode.get("pfsp_pool_size", 0),
                "pfsp_window_size": latest_episode.get("pfsp_window_size"),
                "pfsp_min_episodes": latest_episode.get("pfsp_min_episodes"),
                "pfsp_length_alpha": latest_episode.get("pfsp_length_alpha"),
                "pfsp_temperature": latest_episode.get("pfsp_temperature"),
                "pfsp_min_prob": latest_episode.get("pfsp_min_prob"),
                "pfsp_total_learned_episodes": latest_episode.get("pfsp_total_learned_episodes"),
                "pfsp_window_episodes_by_opponent": latest_episode.get("pfsp_window_episodes_by_opponent", {}),
                "pfsp_window_avg_len_by_opponent": latest_episode.get("pfsp_window_avg_len_by_opponent", {}),
            }
        self.writer.write_event(status)


class DebugCallback(BaseCallback):
    """Callback for logging custom metrics during training."""

    def __init__(self, env, render_freq=10000, n_episodes=1, log_freq=10000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.termination_reasons = deque(maxlen=1000)
        self.env_to_render = env
        self.render_freq = render_freq
        self.n_episodes = n_episodes
        self.distance_mean = deque(maxlen=1000)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', None)
        dones = self.locals.get('dones', None)

        if infos is not None and dones is not None:
            for done, info in zip(dones, infos):
                if done and info is not None and 'done_reason' in info:
                    self.termination_reasons.append(info['done_reason'])
                    if 'distance_mean' in info:
                        self.distance_mean.append(info['distance_mean'])

        if self.num_timesteps % self.log_freq == 0 and self.verbose:
            total = len(self.termination_reasons)
            if total > 0:
                count_hand = sum(1 for r in self.termination_reasons if r in {'Robot Out', 'out of bounds'})
                ratio_hand = count_hand / total
            else:
                ratio_hand = 0.0

            distance_mean = sum(self.distance_mean) / len(self.distance_mean) if len(self.distance_mean) > 0 else 0.0

            self.logger.record("custom/termination_reason_ratio", ratio_hand)
            self.logger.record("custom/distance_mean", distance_mean)
            self.logger.dump(step=self.num_timesteps)

        return True
