# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the probe-lattice scan in SlidingWindowManager.find_longest_cache_hit
(#62336).

The scan probes one block per `sliding_window_contiguous_blocks` from the
right instead of touching every block. These tests pin:
- hit / fallback results stay identical to a reference right-to-left scan
  across randomized cached patterns;
- the lookup count stays O(max_num_blocks / stride) on low-hit-rate rows
  (the point of the change);
- runs that straddle probe boundaries are still found.
"""

import random

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec

pytestmark = pytest.mark.cpu_test


def _make_manager(block_size: int, sliding_window: int, num_blocks: int):
    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )
    block_pool = BlockPool(
        num_gpu_blocks=num_blocks + 10,
        enable_caching=True,
        hash_block_size=block_size,
    )
    manager = SlidingWindowManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
        needs_kv_cache_zeroing=False,
        max_admission_blocks_per_request=10**9,
    )
    return spec, block_pool, manager


def _seed_cache(block_pool, hashes, pattern, group_ids=(0,)):
    block_pool.cached_block_hash_to_block._cache.clear()
    for i, cached in enumerate(pattern):
        if cached:
            for gid in group_ids:
                block_pool.cached_block_hash_to_block.insert(
                    make_block_hash_with_group_id(hashes[i], gid),
                    block_pool.blocks[i + 10],
                )


def _reference_scan(pattern, contiguous_needed):
    """Original right-to-left scan semantics.

    Returns (kept_length, match_found): on a match, the kept length is the
    absolute end of the first (rightmost) run of `contiguous_needed` cached
    blocks; otherwise it is the leftmost run length.
    """
    n = len(pattern)
    num_contig = 0
    for i in range(n - 1, -1, -1):
        if pattern[i]:
            num_contig += 1
            if num_contig >= contiguous_needed:
                return i + num_contig, True
        else:
            num_contig = 0
    return num_contig, False


def test_probe_scan_matches_reference_on_random_patterns():
    rng = random.Random(0)
    for trial in range(200):
        n = rng.randint(0, 40)
        hit_rate = rng.choice([0.0, 0.05, 0.3, 0.7, 1.0])
        pattern = [rng.random() < hit_rate for _ in range(n)]
        block_size, sliding_window = 2, rng.choice([2, 4, 8])
        contiguous = (sliding_window - 1 + block_size - 1) // block_size
        spec, block_pool, manager = _make_manager(block_size, sliding_window, n)
        hashes = [BlockHash(f"{trial}-{i}".encode()) for i in range(n)]
        _seed_cache(block_pool, hashes, pattern)
        blocks, hit_length = manager.find_longest_cache_hit(
            block_hashes=hashes,
            max_length=n * block_size,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=spec,
            drop_eagle_block=False,
            alignment_tokens=block_size,
        )
        expected_len, expected_match = _reference_scan(pattern, contiguous)
        assert hit_length == expected_len * block_size
        if expected_match:
            # The kept prefix runs up to the run end; the run blocks are the
            # cached ones inside it.
            run_end = expected_len
            for i in range(run_end - contiguous, run_end):
                assert blocks[0][i].block_id == i + 10
        assert len(blocks[0]) == expected_len


def test_probe_scan_skips_lookups_on_cold_cache():
    """A cold cache must cost ~max_num_blocks/stride lookups, not n."""
    block_size, sliding_window = 16, 4096
    n = 8192
    spec, block_pool, manager = _make_manager(block_size, sliding_window, n)
    hashes = [BlockHash(str(i).encode()) for i in range(n)]
    _seed_cache(block_pool, hashes, [False] * n)

    lookups = 0
    orig = block_pool.get_cached_block

    def counting(*args, **kwargs):
        nonlocal lookups
        lookups += 1
        return orig(*args, **kwargs)

    block_pool.get_cached_block = counting
    blocks, hit_length = manager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=n * block_size,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=spec,
        drop_eagle_block=False,
        alignment_tokens=block_size,
    )
    block_pool.get_cached_block = orig
    stride = max((sliding_window - 1 + block_size - 1) // block_size, 1)
    assert hit_length == 0
    # One probe per stride, plus at most one trailing leftmost-run lookup.
    assert lookups <= n // stride + 2
    assert lookups < n // 100  # the old scan did n lookups


def test_probe_scan_finds_runs_straddling_probe_boundary():
    """A run longer than the stride that no probe starts must still be found
    through the probe that lands inside it."""
    block_size, sliding_window = 2, 8  # contiguous needed = 4 (stride 4)
    n = 12
    spec, block_pool, manager = _make_manager(block_size, sliding_window, n)
    hashes = [BlockHash(str(i).encode()) for i in range(n)]
    # Blocks 5..8 cached: probes from the right land at 11, 7, 3; probe 7
    # hits inside the run and reconstructs it.
    pattern = [False] * 5 + [True] * 4 + [False] * 3
    _seed_cache(block_pool, hashes, pattern)
    blocks, hit_length = manager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=n * block_size,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=spec,
        drop_eagle_block=False,
        alignment_tokens=block_size,
    )
    expected_len, expected_match = _reference_scan(pattern, 4)
    assert expected_match
    assert hit_length == expected_len * block_size
    assert len(blocks[0]) == expected_len
