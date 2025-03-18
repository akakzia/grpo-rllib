from ray.rllib.algorithms.ppo import PPOConfig
from grpo.grpo_learner import GRPOLearner


def main():
    config = (
        PPOConfig()
        .environment("Pendulum-v1")
        .training(
            learner_class=GRPOLearner,
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
