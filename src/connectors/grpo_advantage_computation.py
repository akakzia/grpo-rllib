from ray.rllib.connectors.connector_v2 import ConnectorV2
import numpy as np
from gymnasium.envs.classic_control.pendulum import angle_normalize
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.postprocessing.zero_padding import unpad_data_if_necessary, split_and_zero_pad_n_episodes
from ray.rllib.connectors.common.numpy_to_tensor import NumpyToTensor
from gymnasium.core import ObsType
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.core.columns import Columns
from gymnasium.spaces.box import Box
from ray.rllib.connectors.learner.general_advantage_estimation import GeneralAdvantageEstimation
import torch


NUM_ACTIONS = 1000


class GRPOAdvantageComputation(GeneralAdvantageEstimation):
    """learner Connector V2 piece computing GRPO advantages on post-processed batches.

    This ConnectorV2: 
    - Uniformly samples random actions using the action space. 
    - Uses the environment's reward function to compute the grpo baseline for current states 
        (grpo_baseline) and next states (next_grpo_baseline).
    - Computes the grpo advantages. 
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
        action_space = episodes[0].action_space
        low = torch.tensor(action_space.low, dtype=torch.float32)
        high = torch.tensor(action_space.high, dtype=torch.float32)

        for batch_key in batch.keys():
            observations = batch[batch_key][Columns.OBS]
            next_observations = batch[batch_key][Columns.NEXT_OBS]
            rewards = batch[batch_key][Columns.REWARDS]
            sample_actions = (high - low) * torch.rand((2, len(observations), NUM_ACTIONS)) + low
            grpo_baseline = self._compute_batch_reward(observations, sample_actions[0])
            next_grpo_baseline = self._compute_batch_reward(next_observations, sample_actions[1])
            grpo_advantages = rewards * 0.99 * next_grpo_baseline - grpo_baseline
            batch[batch_key][Postprocessing.ADVANTAGES] = grpo_advantages

        return batch

    @staticmethod
    def _compute_batch_reward(observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Compute the batch reward as a proxy to the GRPO baseline."""
        cos_th, sin_th, thdot = observations[:, 0], observations[:, 1], observations[:, 2]
        th = np.arctan(sin_th / cos_th)
        th_repeated = th.unsqueeze(1).repeat(1, NUM_ACTIONS)
        thdot_repeated = thdot.unsqueeze(1).repeat(1, NUM_ACTIONS)
        costs = torch.mean(
            angle_normalize(th_repeated) ** 2 + 0.1 * thdot_repeated ** 2 + 0.001 * (actions ** 2), 
            dim=1
        )
        return costs
