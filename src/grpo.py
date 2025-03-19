from ray.rllib.algorithms.ppo import PPOConfig
from src.connectors.add_observations_and_next_observations_to_batch import AddObservationsAndNextObservationsToBatch


class GRPOConfig(PPOConfig):
    def build_learner_connector(self, input_observation_space, input_action_space, device=None):
        """Build GRPO's learner connector.

        Override the first connector by AddObservationsAndNextObservationsToBatch which adds both 
        observations and next observation to the sample batch.
        """
        pipeline = super().build_learner_connector(input_observation_space, input_action_space, device)
        pipeline.connectors[0] = AddObservationsAndNextObservationsToBatch(
            as_learner_connector=True
        )
        return pipeline