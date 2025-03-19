import numpy as np 
from ray.rllib.core.columns import Columns
from ray.rllib.connectors.common.add_observations_from_episodes_to_batch import AddObservationsFromEpisodesToBatch


class AddObservationsAndNextObservationsToBatch(AddObservationsFromEpisodesToBatch):
    def __call__(self, *, rl_module, batch, episodes, explore = None, shared_data = None, **kwargs):
        assert self._as_learner_connector, "Connector only supported as a learner connector."
        if Columns.OBS in batch and Columns.NEXT_OBS in batch:
            return batch

        for sa_episode in self.single_agent_episode_iterator(
            episodes,
            agents_that_stepped_only=not self._as_learner_connector,
        ):
            # For each single agent episode, add observations and next observations items to their 
            # corresponding column.
            obs_size = sa_episode.observation_space.shape[0]
            obs_to_add = sa_episode.get_observations(slice(0, len(sa_episode)))
            next_obs_to_add = np.append(
                sa_episode.get_observations(slice(1, len(sa_episode))), 
                # Dummy placeholder as the last observation does not have a next observation
                np.zeros((1, obs_size), dtype=np.float32), 
                axis=0
            )
            for column, items_to_add in [(Columns.OBS, obs_to_add), (Columns.NEXT_OBS, next_obs_to_add)]:
                self.add_n_batch_items(
                    batch,
                    column,
                    items_to_add=items_to_add,
                    num_items=len(sa_episode)-1,
                    single_agent_episode=sa_episode,
                )
        return batch