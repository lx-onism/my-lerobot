#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
from collections.abc import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EpisodeAwareSampler:
    """Sampler over episode frames that stores only per-episode boundaries.

    Logical positions map to frame indices on the fly (O(num_episodes) construction memory)
    instead of materializing a Python list of every frame index.

    Each epoch is shuffled with a `torch.randperm` seeded from `(seed, epoch)`, so the data order
    is a pure function of `(seed, epoch)`: it reproduces on every rank without synchronizing the
    global RNG (no `generator` to sync across distributed ranks), and `state_dict` /
    `load_state_dict` resume a run sample-exactly by regenerating the epoch's permutation and
    continuing from the saved offset. Each call to `__iter__` advances the epoch. During a
    resumed epoch, `__len__` still reports the full length.

    Epoch advancement: `__iter__` eagerly advances the epoch, and `set_epoch` / `load_state_dict`
    set it explicitly. Within a single run callers should rely on exactly one of these mechanisms,
    not both: advancing the epoch by hand *and* letting `__iter__` auto-advance over the same
    iterations would skip or repeat epochs. The training loop drives it purely through `__iter__`
    (via `cycle`); `set_epoch` / `load_state_dict` are used only to (re)position before iteration
    starts (e.g. on resume or in tests).
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
    ):
        """
        Args:
            dataset_from_indices: Start index of each episode in the dataset.
            dataset_to_indices: End index of each episode in the dataset.
            episode_indices_to_use: Episode indices to use; None means all.
            drop_n_first_frames: Frames to drop from the start of each episode.
            drop_n_last_frames: Frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            seed: Seed the permutation is derived from (together with the epoch).
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError(
                f"dataset_from_indices and dataset_to_indices must have the same length, "
                f"got {len(from_indices)} and {len(to_indices)}"
            )

        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(from_indices), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        starts = from_indices + drop_n_first_frames
        lengths = to_indices - drop_n_last_frames - starts
        for episode_idx in np.flatnonzero(used & (lengths <= 0)):
            logger.warning(
                "Episode %d has %d frames but drop_n_first_frames=%d and "
                "drop_n_last_frames=%d removes all frames. Skipping.",
                episode_idx,
                to_indices[episode_idx] - from_indices[episode_idx],
                drop_n_first_frames,
                drop_n_last_frames,
            )
        used &= lengths > 0
        if not used.any():
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self._starts = starts[used]
        self._cum_lengths = np.cumsum(lengths[used])
        self._num_frames = int(self._cum_lengths[-1])
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._start_index = 0
        self._absolute_to_relative = absolute_to_relative_idx

    @property
    def indices(self) -> list[int]:
        """Materialized frame indices in unshuffled order; O(num_frames), introspection only."""
        return [self._frame_index(k) for k in range(self._num_frames)]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        # Derive a per-epoch seed from (seed, epoch) so the permutation is a pure function of both
        # and reproduces identically on every rank without touching the global RNG.
        epoch_seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(epoch_seed)

    def _frame_index(self, position: int) -> int:
        episode = int(np.searchsorted(self._cum_lengths, position, side="right"))
        position_in_episode = position - (int(self._cum_lengths[episode - 1]) if episode > 0 else 0)
        absolute_idx = int(self._starts[episode]) + position_in_episode
        if self._absolute_to_relative is not None:
            return self._absolute_to_relative[absolute_idx]
        return absolute_idx

    def __iter__(self) -> Iterator[int]:
        # Advance epoch state eagerly, not on first consumption of the generator.
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        if self.shuffle:
            order = torch.randperm(self._num_frames, generator=self._epoch_generator(epoch))
            for k in range(start, self._num_frames):
                yield self._frame_index(int(order[k]))
        else:
            for k in range(start, self._num_frames):
                yield self._frame_index(k)

    def __len__(self) -> int:
        return self._num_frames


class TaskBalancedEpisodeAwareSampler:
    """Oversample task groups so every task contributes equally in each epoch.

    Frames are interleaved task-by-task (task A, B, C, A, B, C, ...). Smaller tasks are
    sampled with replacement to reach the largest task's frame count. The resulting order
    is deterministic from ``seed`` and ``epoch``, and supports the same checkpoint resume
    state format as :class:`EpisodeAwareSampler`.
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_tasks: list[list[str]],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
    ):
        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError(
                "dataset_from_indices and dataset_to_indices must have the same length, "
                f"got {len(from_indices)} and {len(to_indices)}"
            )
        if len(episode_tasks) != len(from_indices):
            raise ValueError(
                f"episode_tasks has {len(episode_tasks)} entries but {len(from_indices)} episode bounds"
            )

        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used[:] = False
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        task_frames: dict[str, list[int]] = {}
        for episode_idx in np.flatnonzero(used):
            start = int(from_indices[episode_idx]) + drop_n_first_frames
            end = int(to_indices[episode_idx]) - drop_n_last_frames
            if end <= start:
                logger.warning(
                    "Episode %d has %d frames but drop_n_first_frames=%d and "
                    "drop_n_last_frames=%d removes all frames. Skipping.",
                    int(episode_idx),
                    int(to_indices[episode_idx] - from_indices[episode_idx]),
                    drop_n_first_frames,
                    drop_n_last_frames,
                )
                continue

            tasks = episode_tasks[episode_idx]
            task = tasks[0] if tasks else ""
            task_frames.setdefault(task, []).extend(range(start, end))

        if not task_frames:
            raise ValueError("No valid task frames remain after filtering or frame drops.")

        self._task_names = list(task_frames)
        self._task_frames = [task_frames[name] for name in self._task_names]
        self._target_frames_per_task = max(len(frames) for frames in self._task_frames)
        self._num_frames = self._target_frames_per_task * len(self._task_names)
        self.seed = seed
        self._epoch = 0
        self._start_index = 0
        self._absolute_to_relative = absolute_to_relative_idx

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int, task_idx: int) -> torch.Generator:
        epoch_seed = int(np.random.SeedSequence([self.seed, epoch, task_idx]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(epoch_seed)

    def __iter__(self) -> Iterator[int]:
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        permutations = []
        for task_idx, frames in enumerate(self._task_frames):
            order = torch.randperm(len(frames), generator=self._epoch_generator(epoch, task_idx)).tolist()
            permutations.append([frames[position] for position in order])

        for position in range(start, self._num_frames):
            task_idx = position % len(self._task_names)
            frame_idx = (position // len(self._task_names)) % len(permutations[task_idx])
            absolute_idx = permutations[task_idx][frame_idx]
            yield self._absolute_to_relative[absolute_idx] if self._absolute_to_relative else absolute_idx

    @property
    def indices(self) -> list[int]:
        return list(self._iter_epoch(self._epoch, 0))

    @property
    def task_names(self) -> list[str]:
        return list(self._task_names)

    @property
    def target_frames_per_task(self) -> int:
        return self._target_frames_per_task

    def __len__(self) -> int:
        return self._num_frames


def compute_sampler_state(step: int, num_frames: int, batch_size: int, num_processes: int) -> dict:
    """Map an optimization step to an `EpisodeAwareSampler` state for sample-exact resume.

    Under accelerate's batch sharding, one step consumes `batch_size * num_processes` sampler
    positions and each rank sees `ceil(ceil(num_frames / batch_size) / num_processes)` batches
    per epoch (`even_batches` padding included). The start index provably stays below
    `num_frames`; the `min` is defensive.

    Assumptions (resume is only sample-exact when they hold):
        - `num_processes` and `batch_size` match the run that wrote the checkpoint. Both scale how
          many positions a step consumes, so the epoch/offset are wrong if either changed. The
          caller passes the checkpoint's `num_processes` and `batch_size` and warns on a mismatch.
        - accelerate uses `even_batches=True` (its default). The `ceil(... / num_processes)` term
          mirrors that padding; with `even_batches=False` the per-epoch batch count differs and
          the boundary is off.
    """
    batches_per_epoch = math.ceil(math.ceil(num_frames / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    start_index = min(batches_into_epoch * batch_size * num_processes, num_frames)
    return {"epoch": epoch, "start_index": start_index}
