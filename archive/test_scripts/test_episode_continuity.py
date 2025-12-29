"""
简化测试：验证Episode连续性和Memory重置

这个测试专注于验证最核心的问题：
1. Dataset能否正确加载zarr和MD-Map GT数据
2. EpisodeSequentialSampler是否保证batch内帧连续
3. 能否正确识别episode边界

不包含完整的模型训练，只验证数据流和episode逻辑。

Author: Claude Code
Date: 2025-12-18
"""

import sys
from pathlib import Path
import numpy as np
import torch
import zarr

# Add UniT to path
sys.path.insert(0, str(Path(__file__).parent))

from UniT.dataset.episode_sampler import EpisodeSequentialSampler


def test_data_loading():
    """测试1: 验证zarr和MD-Map GT数据可以正确加载"""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)

    # Check zarr data
    zarr_path = Path("data/test_replay_buffer.zarr")
    if not zarr_path.exists():
        print(f"❌ Zarr file not found: {zarr_path}")
        return False

    root = zarr.open(str(zarr_path), mode='r')

    print(f"\nZarr structure:")
    print(f"  data/tactile_image: {root['data']['tactile_image'].shape}")
    print(f"  meta/episode_ends: {root['meta']['episode_ends'][:].tolist()}")

    tactile_images = root['data']['tactile_image']
    episode_ends = root['meta']['episode_ends'][:]

    assert tactile_images.shape == (265, 240, 320, 3), f"Wrong tactile shape: {tactile_images.shape}"
    assert episode_ends.tolist() == [84, 188, 265], f"Wrong episode_ends: {episode_ends.tolist()}"

    print(f"✓ Zarr data loaded correctly")

    # Check MD-Map GT
    mdmap_root = Path("data/test_mdmap_gt")
    if not mdmap_root.exists():
        print(f"❌ MD-Map GT not found: {mdmap_root}")
        return False

    for ep_idx in range(3):
        npz_path = mdmap_root / f"episode_{ep_idx:03d}" / "mdmap_gt.npz"
        if not npz_path.exists():
            print(f"❌ Missing MD-Map GT: {npz_path}")
            return False

        data = np.load(npz_path)
        md_map = data['md_map']

        print(f"  Episode {ep_idx}: {npz_path.name} shape={md_map.shape}")

        # Verify shape matches expected
        expected_frames = [84, 104, 77][ep_idx]
        assert md_map.shape[0] == expected_frames, f"Frame count mismatch: {md_map.shape[0]} vs {expected_frames}"
        assert md_map.shape[1:] == (240, 320, 3), f"MD-Map shape wrong: {md_map.shape[1:]}"

    print(f"✓ All MD-Map GT files loaded correctly")
    print("✅ TEST 1 PASSED\n")
    return True


def test_episode_sampler():
    """测试2: 验证EpisodeSequentialSampler保证batch连续性"""
    print("="*60)
    print("TEST 2: Episode Sequential Sampler")
    print("="*60)

    episode_ends = np.array([84, 188, 265])
    batch_size = 8

    sampler = EpisodeSequentialSampler(
        episode_ends=episode_ends,
        batch_size=batch_size,
        shuffle_episodes=True,
    )

    print(f"\nSampler config:")
    print(f"  Episode ends: {episode_ends.tolist()}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches: {len(sampler)}")

    # Collect all batches
    batches = list(sampler)

    print(f"\nVerifying batch continuity...")

    # Check each batch
    for i, batch in enumerate(batches):
        # Find which episode this batch belongs to
        batch_start = batch[0]
        episode_id = None
        for ep_idx, end in enumerate(episode_ends):
            start = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
            if start <= batch_start < end:
                episode_id = ep_idx
                break

        # Verify all indices in batch are from same episode
        for idx in batch:
            ep_check = None
            for ep_idx, end in enumerate(episode_ends):
                start = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
                if start <= idx < end:
                    ep_check = ep_idx
                    break

            if ep_check != episode_id:
                print(f"❌ Batch {i}: Mixed episodes! {episode_id} vs {ep_check}")
                return False

        # Verify indices are consecutive
        if len(batch) > 1:
            diffs = [batch[j+1] - batch[j] for j in range(len(batch)-1)]
            if not all(d == 1 for d in diffs):
                print(f"❌ Batch {i}: Not consecutive! {batch}")
                return False

        if i % 10 == 0 or i < 5:
            print(f"  Batch {i}: Episode {episode_id}, indices {batch[0]}-{batch[-1]} (size={len(batch)})")

    print(f"\n✓ All {len(batches)} batches verified:")
    print(f"  - Each batch from single episode")
    print(f"  - Indices consecutive within batch")
    print("✅ TEST 2 PASSED\n")
    return True


def test_episode_boundary_detection():
    """测试3: 验证能否正确检测episode边界"""
    print("="*60)
    print("TEST 3: Episode Boundary Detection")
    print("="*60)

    episode_ends = np.array([84, 188, 265])
    batch_size = 16  # Larger batch to test boundary crossing

    sampler = EpisodeSequentialSampler(
        episode_ends=episode_ends,
        batch_size=batch_size,
        shuffle_episodes=False,  # Don't shuffle for this test
    )

    batches = list(sampler)

    print(f"\nChecking episode transitions...")
    prev_episode = -1
    transition_count = 0

    for i, batch in enumerate(batches):
        batch_start = batch[0]

        # Determine episode
        episode_id = None
        for ep_idx, end in enumerate(episode_ends):
            start = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
            if start <= batch_start < end:
                episode_id = ep_idx
                break

        # Check if episode changed
        if episode_id != prev_episode and prev_episode != -1:
            print(f"  Batch {i}: Episode transition {prev_episode} → {episode_id}")
            print(f"    Last batch of ep {prev_episode}: {batches[i-1]}")
            print(f"    First batch of ep {episode_id}: {batch}")
            transition_count += 1

        prev_episode = episode_id

    expected_transitions = len(episode_ends) - 1  # Should be 2 transitions (0→1, 1→2)
    assert transition_count == expected_transitions, \
        f"Wrong number of transitions: {transition_count} vs {expected_transitions}"

    print(f"\n✓ Detected {transition_count} episode transitions correctly")
    print("✅ TEST 3 PASSED\n")
    return True


def test_memory_reset_simulation():
    """测试4: 模拟Memory重置逻辑"""
    print("="*60)
    print("TEST 4: Memory Reset Simulation")
    print("="*60)

    episode_ends = np.array([84, 188, 265])
    batch_size = 8

    sampler = EpisodeSequentialSampler(
        episode_ends=episode_ends,
        batch_size=batch_size,
        shuffle_episodes=True,
    )

    batches = list(sampler)

    print(f"\nSimulating training loop with memory resets...")

    # Simulate a memory module state
    class FakeMemory:
        def __init__(self):
            self.queue = []
            self.reset_count = 0

        def reset(self):
            self.queue = []
            self.reset_count += 1

        def update(self, frame_idx):
            self.queue.append(frame_idx)
            if len(self.queue) > 5:  # K=5
                self.queue.pop(0)

    memory = FakeMemory()
    last_episode_id = None

    for batch_idx, batch in enumerate(batches):
        # Determine episode ID
        batch_start = batch[0]
        episode_id = None
        for ep_idx, end in enumerate(episode_ends):
            start = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
            if start <= batch_start < end:
                episode_id = ep_idx
                break

        # Check if episode changed → reset memory
        if last_episode_id is not None and episode_id != last_episode_id:
            memory.reset()
            if batch_idx < 20:  # Print first few resets
                print(f"  Batch {batch_idx}: Memory RESET (ep {last_episode_id} → {episode_id})")

        # Update memory with batch frames
        for frame_idx in batch:
            memory.update(frame_idx)

        last_episode_id = episode_id

    expected_resets = len(episode_ends) - 1  # 2 resets for 3 episodes
    assert memory.reset_count == expected_resets, \
        f"Wrong number of resets: {memory.reset_count} vs {expected_resets}"

    print(f"\n✓ Memory reset {memory.reset_count} times (correct)")
    print(f"✓ Final memory queue: {memory.queue[-5:]}")
    print("✅ TEST 4 PASSED\n")
    return True


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# Episode Continuity & Memory Reset Test")
    print("#"*60)

    all_passed = True

    try:
        all_passed &= test_data_loading()
        all_passed &= test_episode_sampler()
        all_passed &= test_episode_boundary_detection()
        all_passed &= test_memory_reset_simulation()

        if all_passed:
            print("\n" + "#"*60)
            print("# 🎉 ALL TESTS PASSED!")
            print("#"*60)
            print("\n核心功能验证:")
            print("  ✅ 数据加载正确 (zarr + MD-Map GT)")
            print("  ✅ Batch内帧连续 (同一episode)")
            print("  ✅ Episode边界检测准确")
            print("  ✅ Memory重置逻辑正确")
            print("\n结论:")
            print("  EpisodeSequentialSampler可以解决Memory模块的时序连续性问题")
            print("  现在可以安全地集成到完整训练流程中")
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
