"""A small, dependency-free simulation of a paged KV-cache manager.

This module manages metadata only; it does not calculate attention or allocate real
GPU tensors.  A physical block stands in for the K/V tensors held for every model
layer.  The implementation is intentionally small enough to discuss in an
interview while retaining the important production concepts:

* logical session block tables separated from physical blocks;
* full-block prefix reuse;
* reference counting and copy-on-write for conversation branches;
* an LRU cache for inactive prefix blocks; and
* admission failure when all physical blocks are active.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


PrefixKey = Tuple[str, int, Tuple[int, ...]]


class CacheCapacityError(RuntimeError):
    """Raised when no free or evictable physical KV block exists."""


@dataclass(frozen=True)
class KVCacheConfig:
    block_size: int = 16
    num_gpu_blocks: int = 1024
    max_context_tokens: int = 4096
    model_namespace: str = "model-v1"
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype_bytes: int = 2
    tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        positive_fields = (
            "block_size",
            "num_gpu_blocks",
            "max_context_tokens",
            "num_layers",
            "num_kv_heads",
            "head_dim",
            "dtype_bytes",
            "tensor_parallel_size",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_kv_heads % self.tensor_parallel_size != 0:
            raise ValueError("num_kv_heads must be divisible by tensor_parallel_size")

    @property
    def bytes_per_block_per_gpu(self) -> int:
        local_kv_heads = self.num_kv_heads // self.tensor_parallel_size
        # 2 means one K tensor plus one V tensor.
        return (
            2
            * self.num_layers
            * self.block_size
            * local_kv_heads
            * self.head_dim
            * self.dtype_bytes
        )


@dataclass
class CacheMetrics:
    prefix_hit_tokens: int = 0
    computed_tokens: int = 0
    allocations: int = 0
    evictions: int = 0
    cow_copies: int = 0


@dataclass(frozen=True)
class CacheOperationResult:
    reused_tokens: int
    computed_tokens: int
    block_ids: Tuple[int, ...]


@dataclass
class PhysicalBlock:
    block_id: int
    token_ids: List[int]
    previous_hash: int
    content_hash: int
    kv_bytes: int
    ref_count: int = 0
    last_access: int = 0
    cache_key: Optional[PrefixKey] = None

    @property
    def is_full(self) -> bool:
        return self.cache_key is not None


@dataclass
class SessionState:
    session_id: str
    token_ids: List[int] = field(default_factory=list)
    block_ids: List[int] = field(default_factory=list)


class KVCacheManager:
    """Manage session block tables over a fixed-size physical block pool."""

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.metrics = CacheMetrics()
        self._clock = 0
        self._free_block_ids = list(reversed(range(config.num_gpu_blocks)))
        self._blocks: Dict[int, PhysicalBlock] = {}
        self._prefix_index: Dict[PrefixKey, int] = {}
        self._sessions: Dict[str, SessionState] = {}

    def create_session(
        self, session_id: str, token_ids: Iterable[int]
    ) -> CacheOperationResult:
        if session_id in self._sessions:
            raise ValueError(f"session already exists: {session_id}")
        tokens = list(token_ids)
        self._validate_context(tokens)

        matched_ids, matched_tokens, previous_hash = self._match_prefix(tokens)
        acquired_ids: List[int] = []
        try:
            for block_id in matched_ids:
                self._acquire(block_id)
                acquired_ids.append(block_id)

            remaining = tokens[matched_tokens:]
            required_blocks = math.ceil(len(remaining) / self.config.block_size)
            self._reserve_blocks(required_blocks)

            block_ids = list(acquired_ids)
            for offset in range(0, len(remaining), self.config.block_size):
                page = remaining[offset : offset + self.config.block_size]
                block = self._allocate(page, previous_hash)
                self._acquire(block.block_id)
                block_ids.append(block.block_id)
                previous_hash = block.content_hash
        except Exception:
            for block_id in acquired_ids:
                self._release(block_id)
            raise

        self._sessions[session_id] = SessionState(session_id, tokens, block_ids)
        computed_tokens = len(tokens) - matched_tokens
        self.metrics.prefix_hit_tokens += matched_tokens
        self.metrics.computed_tokens += computed_tokens
        return CacheOperationResult(matched_tokens, computed_tokens, tuple(block_ids))

    def append_tokens(
        self, session_id: str, token_ids: Iterable[int]
    ) -> CacheOperationResult:
        session = self._get_session(session_id)
        new_tokens = list(token_ids)
        self._validate_context(session.token_ids + new_tokens)
        if not new_tokens:
            return CacheOperationResult(0, 0, tuple(session.block_ids))

        tail_space = 0
        copy_tail = False
        if session.block_ids:
            tail = self._blocks[session.block_ids[-1]]
            if len(tail.token_ids) < self.config.block_size:
                tail_space = self.config.block_size - len(tail.token_ids)
                copy_tail = tail.ref_count > 1

        tokens_after_tail = max(0, len(new_tokens) - tail_space)
        required_blocks = math.ceil(tokens_after_tail / self.config.block_size)
        if copy_tail:
            required_blocks += 1
        self._reserve_blocks(required_blocks)

        pending = list(new_tokens)
        if tail_space:
            tail = self._blocks[session.block_ids[-1]]
            if copy_tail:
                copied = self._allocate(list(tail.token_ids), tail.previous_hash)
                self._acquire(copied.block_id)
                session.block_ids[-1] = copied.block_id
                self._release(tail.block_id)
                tail = copied
                self.metrics.cow_copies += 1

            amount = min(tail_space, len(pending))
            tail.token_ids.extend(pending[:amount])
            del pending[:amount]
            tail.content_hash = self._hash_block(tail.previous_hash, tail.token_ids)
            self._touch(tail)
            if len(tail.token_ids) == self.config.block_size:
                self._index_full_block(tail)

        previous_hash = (
            self._blocks[session.block_ids[-1]].content_hash
            if session.block_ids
            else 0
        )
        for offset in range(0, len(pending), self.config.block_size):
            page = pending[offset : offset + self.config.block_size]
            block = self._allocate(page, previous_hash)
            self._acquire(block.block_id)
            session.block_ids.append(block.block_id)
            previous_hash = block.content_hash

        session.token_ids.extend(new_tokens)
        self.metrics.computed_tokens += len(new_tokens)
        return CacheOperationResult(0, len(new_tokens), tuple(session.block_ids))

    def fork_session(self, child_session_id: str, parent_session_id: str) -> None:
        if child_session_id in self._sessions:
            raise ValueError(f"session already exists: {child_session_id}")
        parent = self._get_session(parent_session_id)
        for block_id in parent.block_ids:
            self._acquire(block_id)
        self._sessions[child_session_id] = SessionState(
            child_session_id,
            list(parent.token_ids),
            list(parent.block_ids),
        )

    def replace_session_tokens(
        self, session_id: str, token_ids: Iterable[int]
    ) -> CacheOperationResult:
        """Replace edited history, reusing its unchanged complete-block prefix."""
        tokens = list(token_ids)
        self._validate_context(tokens)
        self.release_session(session_id)
        return self.create_session(session_id, tokens)

    def release_session(self, session_id: str) -> None:
        session = self._get_session(session_id)
        for block_id in session.block_ids:
            self._release(block_id)
        del self._sessions[session_id]

    def session_tokens(self, session_id: str) -> Tuple[int, ...]:
        return tuple(self._get_session(session_id).token_ids)

    def session_block_ids(self, session_id: str) -> Tuple[int, ...]:
        return tuple(self._get_session(session_id).block_ids)

    def block_ref_count(self, block_id: int) -> int:
        return self._blocks[block_id].ref_count

    def stats(self) -> Dict[str, int]:
        active = sum(block.ref_count > 0 for block in self._blocks.values())
        cached = sum(block.ref_count == 0 for block in self._blocks.values())
        return {
            "sessions": len(self._sessions),
            "active_blocks": active,
            "inactive_cached_blocks": cached,
            "free_blocks": len(self._free_block_ids),
            "bytes_per_block_per_gpu": self.config.bytes_per_block_per_gpu,
            "prefix_hit_tokens": self.metrics.prefix_hit_tokens,
            "computed_tokens": self.metrics.computed_tokens,
            "evictions": self.metrics.evictions,
            "cow_copies": self.metrics.cow_copies,
        }

    def _match_prefix(self, tokens: List[int]) -> Tuple[List[int], int, int]:
        matched: List[int] = []
        matched_tokens = 0
        previous_hash = 0
        size = self.config.block_size
        for offset in range(0, len(tokens) - size + 1, size):
            page = tuple(tokens[offset : offset + size])
            key = (self.config.model_namespace, previous_hash, page)
            block_id = self._prefix_index.get(key)
            if block_id is None:
                break
            block = self._blocks[block_id]
            matched.append(block_id)
            matched_tokens += size
            previous_hash = block.content_hash
        return matched, matched_tokens, previous_hash

    def _allocate(self, token_ids: List[int], previous_hash: int) -> PhysicalBlock:
        if not self._free_block_ids:
            self._evict_one()
        block_id = self._free_block_ids.pop()
        block = PhysicalBlock(
            block_id=block_id,
            token_ids=list(token_ids),
            previous_hash=previous_hash,
            content_hash=self._hash_block(previous_hash, token_ids),
            kv_bytes=self.config.bytes_per_block_per_gpu,
        )
        self._blocks[block_id] = block
        self.metrics.allocations += 1
        self._touch(block)
        if len(token_ids) == self.config.block_size:
            self._index_full_block(block)
        return block

    def _index_full_block(self, block: PhysicalBlock) -> None:
        key = (
            self.config.model_namespace,
            block.previous_hash,
            tuple(block.token_ids),
        )
        # If an identical immutable block already exists, keep it as the canonical
        # prefix entry. The duplicate active block remains valid but is not cached
        # after its final reference is released.
        if key not in self._prefix_index:
            self._prefix_index[key] = block.block_id
            block.cache_key = key

    def _acquire(self, block_id: int) -> None:
        block = self._blocks[block_id]
        block.ref_count += 1
        self._touch(block)

    def _release(self, block_id: int) -> None:
        block = self._blocks[block_id]
        if block.ref_count <= 0:
            raise RuntimeError(f"block {block_id} has no active reference")
        block.ref_count -= 1
        self._touch(block)
        if block.ref_count == 0 and block.cache_key is None:
            self._remove_block(block)

    def _reserve_blocks(self, count: int) -> None:
        while len(self._free_block_ids) < count:
            self._evict_one()

    def _evict_one(self) -> None:
        candidates = [block for block in self._blocks.values() if block.ref_count == 0]
        if not candidates:
            raise CacheCapacityError(
                "KV cache is full: all physical blocks are referenced by active sessions"
            )
        victim = min(candidates, key=lambda block: block.last_access)
        self._remove_block(victim)
        self.metrics.evictions += 1

    def _remove_block(self, block: PhysicalBlock) -> None:
        if block.cache_key is not None:
            if self._prefix_index.get(block.cache_key) == block.block_id:
                del self._prefix_index[block.cache_key]
        del self._blocks[block.block_id]
        self._free_block_ids.append(block.block_id)

    def _touch(self, block: PhysicalBlock) -> None:
        self._clock += 1
        block.last_access = self._clock

    def _hash_block(self, previous_hash: int, token_ids: Iterable[int]) -> int:
        payload = (
            f"{self.config.model_namespace}|{previous_hash}|"
            + ",".join(str(token) for token in token_ids)
        ).encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")

    def _validate_context(self, token_ids: List[int]) -> None:
        if len(token_ids) > self.config.max_context_tokens:
            raise ValueError(
                f"context has {len(token_ids)} tokens, maximum is "
                f"{self.config.max_context_tokens}"
            )
        if any(not isinstance(token, int) or token < 0 for token in token_ids):
            raise ValueError("token_ids must contain non-negative integers")

    def _get_session(self, session_id: str) -> SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown session: {session_id}") from exc
