from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_learner import PPOLearner
from ray.rllib.connectors.connector_v2 import ConnectorV2
import numpy as np


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
            explore = None, 
            shared_data = None, 
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
            observations = episode.observations[:-1] # List[ObsType] Of length batch size - 1 
            observation_sample_actions = np.array(
                [episode.action_space.sample() for _ in observations for _ in range(NUM_ACTIONS)]
                ).reshape(len(observations), NUM_ACTIONS)
            next_observations = episode.observations[1:] # List[ObsType] Of length batch size - 1 
        


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