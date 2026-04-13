"""Training callbacks for reinforcement learning."""

from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


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
            log = self.model.logger.name_to_value
            ep_rew = log.get('rollout/ep_rew_mean', None)
            ep_len = log.get('rollout/ep_len_mean', None)
            loss = log.get('train/loss', None)
            v_loss = log.get('train/value_loss', None)
            p_loss = log.get('train/policy_gradient_loss', None)
            ent_loss = log.get('train/entropy_loss', None)
            kl = log.get('train/approx_kl', None)

            total = len(self.termination_reasons)
            if total > 0:
                count_hand = sum(1 for r in self.termination_reasons if r == 'out of bounds')
                ratio_hand = count_hand / total
            else:
                ratio_hand = 0.0

            distance_mean = sum(self.distance_mean) / len(self.distance_mean) if len(self.distance_mean) > 0 else 0.0

            self.logger.record("custom/termination_reason_ratio", ratio_hand)
            self.logger.record("custom/distance_mean", distance_mean)
            self.logger.dump(step=self.num_timesteps)

        return True
