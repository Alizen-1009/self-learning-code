"""一个适合面试讲解的 Paged KV Cache 元数据实现。

这里的 Block 只保存 token，真实项目中它还会对应 GPU 上的 K/V 张量。
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


class CacheFullError(RuntimeError):
    pass


@dataclass
class Block:
    block_id: int
    token_ids: List[int]
    ref_count: int = 0
    last_used: int = 0


@dataclass
class Session:
    token_ids: List[int] = field(default_factory=list)
    block_ids: List[int] = field(default_factory=list)


class KVCache:
    """固定数量物理 block 上的多 session KV Cache。"""

    def __init__(self, block_size: int = 16, num_blocks: int = 1024):
        if block_size <= 0 or num_blocks <= 0:
            raise ValueError("block_size and num_blocks must be positive")
        self.block_size = block_size
        self.blocks: Dict[int, Block] = {}
        self.free_ids = list(range(num_blocks))
        self.sessions: Dict[str, Session] = {}
        # 完整 token 前缀 -> 物理 block。真实实现通常使用 prefix hash。
        self.prefix_index: Dict[Tuple[int, ...], int] = {}
        self.clock = 0

    def create(self, session_id: str, token_ids: Iterable[int]) -> Tuple[int, ...]:
        if session_id in self.sessions:
            raise ValueError("session already exists")

        tokens = list(token_ids)
        blocks: List[int] = []
        reused = 0

        # 只复用完整前缀 block，最后一个不完整 block 不进入前缀索引。
        for end in range(self.block_size, len(tokens) + 1, self.block_size):
            prefix = tuple(tokens[:end])
            block_id = self.prefix_index.get(prefix)
            if block_id is None:
                break
            self._acquire(block_id)
            blocks.append(block_id)
            reused = end

        try:
            for start in range(reused, len(tokens), self.block_size):
                page = tokens[start : start + self.block_size]
                prefix = tuple(tokens[: start + len(page)])
                block = self._allocate(page, prefix if len(page) == self.block_size else None)
                self._acquire(block.block_id)
                blocks.append(block.block_id)
        except Exception:
            for block_id in blocks:
                self._release(block_id)
            raise

        self.sessions[session_id] = Session(tokens, blocks)
        return tuple(blocks)

    def append(self, session_id: str, token_ids: Iterable[int]) -> None:
        session = self._session(session_id)
        new_tokens = list(token_ids)
        if not new_tokens:
            return

        # 尾 block 未满时可以原地写；共享尾 block 则先复制（COW）。
        if session.block_ids:
            tail = self.blocks[session.block_ids[-1]]
            if len(tail.token_ids) < self.block_size:
                if tail.ref_count > 1:
                    copied = self._allocate(list(tail.token_ids), None)
                    self._acquire(copied.block_id)
                    session.block_ids[-1] = copied.block_id
                    self._release(tail.block_id)
                    tail = copied

                amount = min(self.block_size - len(tail.token_ids), len(new_tokens))
                added = new_tokens[:amount]
                tail.token_ids.extend(added)
                session.token_ids.extend(added)
                del new_tokens[:amount]
                self._touch(tail)

                if len(tail.token_ids) == self.block_size:
                    self.prefix_index.setdefault(tuple(session.token_ids), tail.block_id)

        while new_tokens:
            page = new_tokens[: self.block_size]
            del new_tokens[: self.block_size]
            all_tokens = session.token_ids + page
            prefix = tuple(all_tokens) if len(page) == self.block_size else None
            block = self._allocate(page, prefix)
            self._acquire(block.block_id)
            session.block_ids.append(block.block_id)
            session.token_ids.extend(page)

    def fork(self, child_id: str, parent_id: str) -> None:
        if child_id in self.sessions:
            raise ValueError("session already exists")
        parent = self._session(parent_id)
        for block_id in parent.block_ids:
            self._acquire(block_id)
        self.sessions[child_id] = Session(list(parent.token_ids), list(parent.block_ids))

    def release(self, session_id: str) -> None:
        session = self._session(session_id)
        for block_id in session.block_ids:
            self._release(block_id)
        del self.sessions[session_id]

    def get_blocks(self, session_id: str) -> Tuple[int, ...]:
        return tuple(self._session(session_id).block_ids)

    def _allocate(self, token_ids: List[int], prefix: Tuple[int, ...] | None) -> Block:
        if not self.free_ids:
            self._evict_one()
        block = Block(self.free_ids.pop(), list(token_ids))
        self.blocks[block.block_id] = block
        self._touch(block)
        if prefix is not None:
            self.prefix_index.setdefault(prefix, block.block_id)
        return block

    def _acquire(self, block_id: int) -> None:
        block = self.blocks[block_id]
        block.ref_count += 1
        self._touch(block)

    def _release(self, block_id: int) -> None:
        block = self.blocks[block_id]
        block.ref_count -= 1
        self._touch(block)

    def _evict_one(self) -> None:
        candidates = [b for b in self.blocks.values() if b.ref_count == 0]
        if not candidates:
            raise CacheFullError("all blocks are still in use")
        victim = min(candidates, key=lambda b: b.last_used)
        for prefix, block_id in list(self.prefix_index.items()):
            if block_id == victim.block_id:
                del self.prefix_index[prefix]
        del self.blocks[victim.block_id]
        self.free_ids.append(victim.block_id)

    def _touch(self, block: Block) -> None:
        self.clock += 1
        block.last_used = self.clock

    def _session(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            raise KeyError(f"unknown session: {session_id}")
        return self.sessions[session_id]
