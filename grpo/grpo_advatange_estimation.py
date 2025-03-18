from ray.rllib.connectors.connector_v2 import ConnectorV2
import numpy as np
from gymnasium.envs.classic_control.pendulum import angle_normalize
from ray.rllib.utils.postprocessing.value_predictions import compute_value_targets
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.postprocessing.zero_padding import split_and_zero_pad_n_episodes
from ray.rllib.connectors.common.numpy_to_tensor import NumpyToTensor
from gymnasium.core import ObsType


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
        module_id = 0
        # Extract all single-agent episodes.
        sa_episodes_list = list(
            self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False)
        )
        # Collect (single-agent) episode lengths for this particular module.
        episode_lens = [
            len(e) for e in sa_episodes_list if e.module_id in [None, module_id]
        ]
        # Compute group relative baseline 
        for episode in episodes:
            episode_lens
            # Temporary hack:
            module = rl_module[module_id]

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


            next_grb_augmented = np.concatenate([0, next_group_relative_baseline])

            module_value_targets = module_value_targets = compute_value_targets()

            module_advantages = module_value_targets - group_relative_baseline 
            # Drop vf-preds, not needed in loss. Note that in the PPORLModule, vf-preds
            # are recomputed with each `forward_train` call anyway.
            # Standardize advantages (used for more stable and better weighted
            # policy gradient computations).
            module_advantages = (module_advantages - module_advantages.mean()) / max(
                1e-4, module_advantages.std()
            )

            # Zero-pad the new computations, if necessary.
            if module.is_stateful():
                module_advantages = np.stack(
                    split_and_zero_pad_n_episodes(
                        module_advantages,
                        episode_lens=episode_lens,
                        max_seq_len=module.model_config["max_seq_len"],
                    ),
                    axis=0,
                )
                module_value_targets = np.stack(
                    split_and_zero_pad_n_episodes(
                        module_value_targets,
                        episode_lens=episode_lens,
                        max_seq_len=module.model_config["max_seq_len"],
                    ),
                    axis=0,
                )
            batch[module_id][Postprocessing.ADVANTAGES] = module_advantages
            batch[module_id][Postprocessing.VALUE_TARGETS] = module_value_targets

        # Convert all GAE results to tensors.
        if self._numpy_to_tensor_connector is None:
            self._numpy_to_tensor_connector = NumpyToTensor(
                as_learner_connector=True, device=device
            )
        tensor_results = self._numpy_to_tensor_connector(
            rl_module=rl_module,
            batch={
                mid: {
                    Postprocessing.ADVANTAGES: module_batch[Postprocessing.ADVANTAGES],
                    Postprocessing.VALUE_TARGETS: (
                        module_batch[Postprocessing.VALUE_TARGETS]
                    ),
                }
                for mid, module_batch in batch.items()
                if vf_preds[mid] is not None
            },
            episodes=episodes,
        )
        # Move converted tensors back to `batch`.
        for mid, module_batch in tensor_results.items():
            batch[mid].update(module_batch)

        return batch
        ## vf_preds at state s_t: group_relative_baseline
        ## vf_preds at state s_tp: next_group_relative_baseline

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
