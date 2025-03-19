from src.grpo import GRPOConfig
from src.grpo_learner import GRPOTorchLearner


def main():
    config: GRPOConfig = (
        GRPOConfig()
        .env_runners(
            num_env_runners=1,
            rollout_fragment_length=200,
            batch_mode="truncate_episodes"
        )
        .debugging(log_level="INFO")
        .environment("Pendulum-v1")
        .training(
            use_critic=False,
            learner_class=GRPOTorchLearner,
        )
    )
    algo = config.build()
    for i in range(5):
        result = algo.train()
        print(result)


if __name__ == "__main__":
    main()
