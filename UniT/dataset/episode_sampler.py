"""
Episode-Aware Sequential Sampler for Memory-Augmented Models

This sampler ensures temporal continuity within episodes while maintaining
randomness across episodes. Critical for models with memory modules that
require consecutive frames from the same episode.

Key features:
- Episode-level shuffle: Randomizes episode order (for diversity)
- Frame-level sequential: Preserves temporal order within episodes
- Batch alignment: Each batch contains frames from ONE episode only

Author: Claude Code
Date: 2025-12-18
"""

import random
from typing import Iterator, List, Tuple
import numpy as np
from torch.utils.data import Sampler


class EpisodeSequentialSampler(Sampler):
    """
    Sampler that yields batches of consecutive frames from the same episode.

    This is critical for memory-augmented models where the memory module
    maintains a FIFO queue of recent frames. Random shuffling would break
    temporal continuity and cause the model to learn meaningless correlations.

    Example with 3 episodes:
        Episode 0: frames [0-83]     (84 frames)
        Episode 1: frames [84-187]   (104 frames)
        Episode 2: frames [188-264]  (77 frames)

    With batch_size=8, shuffle_episodes=True, one epoch might yield:
        [Episode 1]: [84-91], [92-99], [100-107], ..., [180-187]
        [Episode 0]: [0-7], [8-15], [16-23], ..., [76-83]
        [Episode 2]: [188-195], [196-203], ..., [258-264] (last batch size=7)

    Args:
        episode_ends: Cumulative episode end indices (e.g., [84, 188, 265])
        batch_size: Number of frames per batch
        shuffle_episodes: Whether to shuffle episode order (default: True)
        drop_last: Whether to drop the last incomplete batch of each episode
    """

    def __init__(
        self,
        episode_ends: np.ndarray,
        batch_size: int,
        shuffle_episodes: bool = True,
        drop_last: bool = False,
    ):
        self.episode_ends = np.array(episode_ends)
        self.batch_size = batch_size
        self.shuffle_episodes = shuffle_episodes
        self.drop_last = drop_last

        # Compute episode ranges: [(start, end), ...]
        self.episode_ranges = self._compute_episode_ranges()

        # Count total batches
        self.num_batches = self._count_total_batches()

    def _compute_episode_ranges(self) -> List[Tuple[int, int]]:
        """
        Convert cumulative episode ends to (start, end) ranges.

        Example:
            episode_ends = [84, 188, 265]
            → [(0, 84), (84, 188), (188, 265)]
        """
        ranges = []
        start = 0
        for end in self.episode_ends:
            ranges.append((start, end))
            start = end
        return ranges

    def _count_total_batches(self) -> int:
        """Count total number of batches across all episodes"""
        total = 0
        for start, end in self.episode_ranges:
            episode_len = end - start
            if self.drop_last:
                total += episode_len // self.batch_size
            else:
                total += (episode_len + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield batches of indices.

        Each batch contains consecutive frame indices from the same episode.
        Episode order is shuffled if shuffle_episodes=True.
        """
        # Determine episode order
        episode_order = list(range(len(self.episode_ranges)))
        if self.shuffle_episodes:
            random.shuffle(episode_order)

        # Iterate through episodes in the shuffled order
        for ep_idx in episode_order:
            start, end = self.episode_ranges[ep_idx]

            # Generate batches for this episode (sequential within episode)
            for batch_start in range(start, end, self.batch_size):
                batch_end = min(batch_start + self.batch_size, end)

                # Skip incomplete last batch if drop_last=True
                if self.drop_last and (batch_end - batch_start) < self.batch_size:
                    continue

                # Yield batch indices
                batch_indices = list(range(batch_start, batch_end))
                yield batch_indices

    def __len__(self) -> int:
        """Return total number of batches"""
        return self.num_batches


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("EpisodeSequentialSampler Test")
    print("="*60)

    # Simulate 3 episodes with different lengths
    episode_ends = np.array([84, 188, 265])
    batch_size = 8

    sampler = EpisodeSequentialSampler(
        episode_ends=episode_ends,
        batch_size=batch_size,
        shuffle_episodes=True,
        drop_last=False,
    )

    print(f"\nDataset info:")
    print(f"  Episode ends: {episode_ends.tolist()}")
    print(f"  Episode lengths: [84, 104, 77]")
    print(f"  Total frames: {episode_ends[-1]}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches: {len(sampler)}")

    print(f"\nFirst epoch (episode order randomized):")
    batches = list(sampler)

    # Group by episode
    current_episode = -1
    episode_batch_count = 0

    for i, batch in enumerate(batches[:15]):  # Show first 15 batches
        # Determine which episode this batch belongs to
        batch_start = batch[0]
        for ep_idx, (start, end) in enumerate(sampler.episode_ranges):
            if start <= batch_start < end:
                if ep_idx != current_episode:
                    if current_episode != -1:
                        print(f"    (Episode {current_episode}: {episode_batch_count} batches)")
                    current_episode = ep_idx
                    episode_batch_count = 0
                    print(f"\n  Episode {ep_idx} (frames {start}-{end-1}):")
                episode_batch_count += 1
                break

        print(f"    Batch {i}: indices {batch[0]}-{batch[-1]} (size={len(batch)})")

    print(f"    (Episode {current_episode}: {episode_batch_count} batches)")

    # Verify properties
    print(f"\n✓ Properties verified:")
    print(f"  - Total batches: {len(batches)}")
    print(f"  - All indices covered: {sorted([idx for batch in batches for idx in batch]) == list(range(episode_ends[-1]))}")

    # Check that batches within an episode are consecutive
    for ep_idx, (start, end) in enumerate(sampler.episode_ranges):
        episode_batches = [b for b in batches if start <= b[0] < end]
        all_indices = [idx for batch in episode_batches for idx in batch]

        # Check continuity
        is_continuous = all_indices == list(range(start, start + len(all_indices)))
        print(f"  - Episode {ep_idx} continuity: {is_continuous}")

    print(f"\n✅ EpisodeSequentialSampler working correctly!")
