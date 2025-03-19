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

def compute_value_targets(
    values,
    rewards,
    terminateds,
    truncateds,
    gamma: float,
    lambda_: float,
):
    """Computes value function (vf) targets given vf predictions and rewards.

    Note that advantages can then easily be computed via the formula:
    advantages = targets - vf_predictions
    """
    breakpoint()
    # Force-set all values at terminals (not at truncations!) to 0.0.
    orig_values = flat_values = values * (1.0 - terminateds)

    flat_values = np.append(flat_values, 0.0)
    intermediates = rewards + gamma * (1 - lambda_) * flat_values[1:]
    continues = 1.0 - terminateds

    Rs = []
    last = flat_values[-1]
    for t in reversed(range(intermediates.shape[0])):
        last = intermediates[t] + continues[t] * gamma * lambda_ * last
        Rs.append(last)
        if truncateds[t]:
            last = orig_values[t]

    # Reverse back to correct (time) direction.
    value_targets = np.stack(list(reversed(Rs)), axis=0)

    return value_targets.astype(np.float32)


class GRPOAdvantageEstimation(GeneralAdvantageEstimation):
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
        action_space = episodes[0].action_space
        low = torch.tensor(action_space.low, dtype=torch.float32)
        high = torch.tensor(action_space.high, dtype=torch.float32)
        for batch_key in batch.keys():
            observations = batch[batch_key][Columns.OBS]
            next_observations = batch[batch_key][Columns.NEXT_OBS]
            rewards = batch[batch_key][Columns.REWARDS]
            sample_actions = (high - low) * torch.rand((2, 201, NUM_ACTIONS)) + low
            grpo_baseline = self._compute_batch_reward(observations, sample_actions[0])
            next_grpo_baseline = self._compute_batch_reward(next_observations, sample_actions[1])
            # grpo_baseline = torch.cat([grpo_baseline, torch.tensor([0.0])])
            # next_grpo_baseline = torch.cat([next_grpo_baseline, torch.tensor([0.0])])
            grpo_advantages = rewards * 0.99 * next_grpo_baseline - grpo_baseline # TODO apply termination
            batch[batch_key][Postprocessing.ADVANTAGES] = grpo_advantages
        # # Extract all single-agent episodes.
        # sa_episodes_list = list(
        #     self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False)
        # )
        # # Compute group relative baseline 
        # for module_id in rl_module.keys():
        #     module = rl_module[module_id]

        #     # Collect (single-agent) episode lengths for this particular module.
        #     episode_lens = [
        #         len(e) for e in sa_episodes_list if e.module_id in [None, module_id]
        #     ]
        #     grb_array = []
        #     next_grb_array = []
        #     for episode in episodes:
        #         observations = episode.observations[
        #                     :-1]  # List[ObsType] Of length batch size - 1, Shape=[num_obs, obs_size]
        #         next_observations = episode.observations[
        #                             1:]  # List[ObsType] Of length batch size - 1, Shape=[num_obs, obs_size]

        #         # [num_obs -1, NUM_ACTIONS, action_size]
        #         num_observations = observations.shape[0]
        #         observation_sample_actions = self._get_obs_sampled_actions(num_observations, episode.action_space)

        #         # [num_obs -1, NUM_ACTIONS, action_size]
        #         next_observation_sample_actions = self._get_obs_sampled_actions(num_observations, episode.action_space)

        #         # Shape = [num_obs, 1, obs_size]
        #         observations = np.expand_dims(observations, axis=1)
        #         next_observations = np.expand_dims(next_observations, axis=1)

        #         # Shape = [num_obs, NUM_ACTIONS, obs_size]
        #         observations = np.repeat(observations, NUM_ACTIONS, axis=1)
        #         next_observations = np.repeat(next_observations, NUM_ACTIONS, axis=1)

        #         group_relative_baseline = self._compute_batch_reward(observations, observation_sample_actions)
        #         next_group_relative_baseline = self._compute_batch_reward(next_observations,
        #                                                                 next_observation_sample_actions)

        #         grb_array.append(group_relative_baseline)
        #         next_grb_array.append(next_group_relative_baseline)
                
        #     grb_array = np.concatenate(grb_array, axis=0)
        #     next_grb_array = np.concatenate(next_grb_array, axis=0)
            
        #     next_grb_augmented = np.insert(next_grb_array, 0, 0) 
        #     module_value_targets = compute_value_targets(
        #         values=next_grb_augmented,
        #         rewards=unpad_data_if_necessary(
        #             episode_lens,
        #             convert_to_numpy(batch[module_id][Columns.REWARDS]),
        #         ),
        #         terminateds=unpad_data_if_necessary(
        #             episode_lens,
        #             convert_to_numpy(batch[module_id][Columns.TERMINATEDS]),
        #         ),
        #         truncateds=unpad_data_if_necessary(
        #             episode_lens,
        #             convert_to_numpy(batch[module_id][Columns.TRUNCATEDS]),
        #         ),
        #         gamma=0.99,
        #         lambda_=11,
        #     )

        #     module_advantages = module_value_targets - group_relative_baseline 
        #     # Drop vf-preds, not needed in loss. Note that in the PPORLModule, vf-preds
        #     # are recomputed with each `forward_train` call anyway.
        #     # Standardize advantages (used for more stable and better weighted
        #     # policy gradient computations).
        #     module_advantages = (module_advantages - module_advantages.mean()) / max(
        #         1e-4, module_advantages.std()
        #     )

        #     # Zero-pad the new computations, if necessary.
        #     if module.is_stateful():
        #         module_advantages = np.stack(
        #             split_and_zero_pad_n_episodes(
        #                 module_advantages,
        #                 episode_lens=episode_lens,
        #                 max_seq_len=module.model_config["max_seq_len"],
        #             ),
        #             axis=0,
        #         )
        #         module_value_targets = np.stack(
        #             split_and_zero_pad_n_episodes(
        #                 module_value_targets,
        #                 episode_lens=episode_lens,
        #                 max_seq_len=module.model_config["max_seq_len"],
        #             ),
        #             axis=0,
        #         )
        #     batch[module_id][Postprocessing.ADVANTAGES] = module_advantages
        #     batch[module_id][Postprocessing.VALUE_TARGETS] = module_value_targets

        # # Convert all GAE results to tensors.
        # if self._numpy_to_tensor_connector is None:
        #     self._numpy_to_tensor_connector = NumpyToTensor(
        #         as_learner_connector=True, device=device
        #     )
        # tensor_results = self._numpy_to_tensor_connector(
        #     rl_module=rl_module,
        #     batch={
        #         mid: {
        #             Postprocessing.ADVANTAGES: module_batch[Postprocessing.ADVANTAGES],
        #             Postprocessing.VALUE_TARGETS: (
        #                 module_batch[Postprocessing.VALUE_TARGETS]
        #             ),
        #         }
        #         for mid, module_batch in batch.items()
        #     },
        #     episodes=episodes,
        # )
        # # Move converted tensors back to `batch`.
        # for mid, module_batch in tensor_results.items():
        #     batch[mid].update(module_batch)

        return batch
        ## vf_preds at state s_t: group_relative_baseline
        ## vf_preds at state s_tp: next_group_relative_baseline

    def _compute_batch_reward(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        cos_th, sin_th, thdot = observations[:, 0], observations[:, 1], observations[:, 2]
        th = np.arctan(sin_th / cos_th)
        th_repeated = th.unsqueeze(1).repeat(1, NUM_ACTIONS)
        thdot_repeated = thdot.unsqueeze(1).repeat(1, NUM_ACTIONS)

        costs = torch.mean(
            angle_normalize(th_repeated) ** 2 + 0.1 * thdot_repeated ** 2 + 0.001 * (actions ** 2), 
            dim=1
        )
        return costs

    def _get_obs_sampled_actions(self, num_observations: int, action_space) -> np.array:
        return np.array(
            [action_space.sample() for _ in range(num_observations) for _ in range(NUM_ACTIONS)]
        ).reshape(num_observations, NUM_ACTIONS)
