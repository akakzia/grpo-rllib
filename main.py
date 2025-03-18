from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_learner import PPOLearner
from ray.rllib.connectors.connector_v2 import ConnectorV2
import numpy as np
from gymnasium.envs.classic_control.pendulum import angle_normalize

NUM_ACTIONS = 1000


class GRPOAdvantageEstimation(ConnectorV2):
    """_summary_

    Args:
        ConnectorV2 (_type_): _description_
    """

    def __call__(
            self,
            *,
            rl_module,
            batch,
            episodes,
            explore=None,
            shared_data=None,
            **kwargs
    ):
        # Device to place all GAE result tensors (advantages and value targets) on.
        device = None

        # Extract all single-agent episodes.
        sa_episodes_list = list(
            self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False)
        )

        # Compute group relative baseline 
        for episode in episodes:
            observations = episode.observations[
                           :-1]  # List[ObsType] Of length batch size - 1, Shape=[num_obs, obs_size]
            next_observations = episode.observations[
                                1:]  # List[ObsType] Of length batch size - 1, Shape=[num_obs, obs_size]

            # [num_obs -1, NUM_ACTIONS, action_size]
            observation_sample_actions = self._get_obs_sampled_actions(observations, episode.action_space)

            # [num_obs -1, NUM_ACTIONS, action_size]
            next_observation_sample_actions = self._get_obs_sampled_actions(next_observations, episode.action_space)

            # Shape = [num_obs, 1, obs_size]
            observations = np.expand_dims(observations, axis=1)
            next_observations = np.expand_dims(next_observations, axis=1)

            # Shape = [num_obs, NUM_ACTIONS, obs_size]
            observations = np.repeat(observations, NUM_ACTIONS, axis=1)
            next_observations = np.repeat(next_observations, NUM_ACTIONS, axis=1)

            group_relative_baseline = self._compute_batch_reward(observations, observation_sample_actions)
            next_group_relative_baseline = self._compute_batch_reward(next_observations,
                                                                      next_observation_sample_actions)

    def _compute_batch_reward(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        batch_reward = np.zeros(shape=(len(observations), 1))
        for i, obs, u in enumerate(zip(observations, actions)):
            cos_th, sin_th, thdot = obs[:][0], obs[:][1], obs[:][2]

            th = np.arctan(sin_th / cos_th)

            costs = np.mean(angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2), axis=0)
            batch_reward[i] = costs

        return batch_reward

    def _get_obs_sampled_actions(self, observations: list[ObsType], action_space) -> np.array:
        return np.array(
            [action_space.sample() for _ in observations for _ in range(NUM_ACTIONS)]
        ).reshape(len(observations) - 1, NUM_ACTIONS)


def main():
    config = (
        PPOConfig()
        .environment("Pendulum-v1")
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,  # Still used in our CustomGAE
            vf_loss_coeff=0.5,
            kl_coeff=0.2
        )
        .env_runners(
            num_env_runners=4,
            rollout_fragment_length=200,
            batch_mode="truncate_episodes"
        )
        .debugging(log_level="INFO")
    )

    # 5. Execute Training
    algo = config.build()
    for i in range(1):
        result = algo.train()
        print(f"Iter {i}: Reward={result['env_runners']['episode_return_mean']:.1f}")
        print(f"VF Loss: {result['info']['learner']['default_policy']['custom_vf_loss']:.4f}")


if __name__ == "__main__":
    main()
