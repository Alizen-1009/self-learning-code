"""简洁的 Paged KV Cache Pool，使用 PyTorch 存储实际 K/V。

每个请求有独立页表，逻辑页映射到 pool 中可不连续的物理页。
本例一次追加所有层的 KV；单线程使用，不包含前缀共享和换出。
"""

from dataclasses import dataclass, field

import torch


@dataclass
class Sequence:
    pages: list[int] = field(default_factory=list)
    length: int = 0


class KVPool:
    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ):
        if min(num_pages, page_size, num_layers, num_kv_heads, head_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        self.page_size = page_size
        # 每个物理页存放 page_size 个 token 的所有层 KV。
        shape = (num_layers, num_pages, page_size, num_kv_heads, head_dim)
        self.k_cache = torch.empty(shape, dtype=dtype, device=device)
        self.v_cache = torch.empty_like(self.k_cache)
        self.free_pages = list(reversed(range(num_pages)))
        self.sequences: dict[str, Sequence] = {}

    @property
    def num_free_pages(self) -> int:
        return len(self.free_pages)

    def create(self, request_id: str) -> None:
        if request_id in self.sequences:
            raise ValueError(f"request already exists: {request_id}")
        self.sequences[request_id] = Sequence()

    @torch.no_grad()
    def append(self, request_id: str, k: torch.Tensor, v: torch.Tensor) -> None:
        """追加 prefill 或 decode KV，输入形状均为 [layers, tokens, heads, dim]。"""
        seq = self.sequences[request_id]
        if k.ndim != 4 or v.shape != k.shape:
            raise ValueError("k and v must have the same 4D shape")
        if (k.shape[0], k.shape[2], k.shape[3]) != (
            self.k_cache.shape[0], self.k_cache.shape[3], self.k_cache.shape[4]
        ):
            raise ValueError("layers, heads or head_dim does not match the pool")
        for tensor in (k, v):
            if tensor.device != self.k_cache.device or tensor.dtype != self.k_cache.dtype:
                raise ValueError("input device and dtype must match the pool")

        num_tokens = k.shape[1]
        new_length = seq.length + num_tokens
        required_pages = (new_length + self.page_size - 1) // self.page_size
        extra_pages = required_pages - len(seq.pages)
        # 先检查容量，容量不足时不修改页表和已有 KV。
        if extra_pages > self.num_free_pages:
            raise MemoryError(f"need {extra_pages} new pages, have {self.num_free_pages}")
        seq.pages.extend(self.free_pages.pop() for _ in range(extra_pages))

        copied = 0
        while copied < num_tokens:
            position = seq.length + copied
            logical_page, offset = divmod(position, self.page_size)
            physical_page = seq.pages[logical_page]
            count = min(self.page_size - offset, num_tokens - copied)
            # 跨页时分段写入；未满的尾页会在下一次 append 时继续使用。
            self.k_cache[:, physical_page, offset : offset + count].copy_(
                k[:, copied : copied + count]
            )
            self.v_cache[:, physical_page, offset : offset + count].copy_(
                v[:, copied : copied + count]
            )
            copied += count
        seq.length = new_length

    def block_table(self, request_id: str) -> list[int]:
        """返回逻辑页 -> 物理页映射的副本。"""
        return list(self.sequences[request_id].pages)

    def read(self, request_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """按 token 顺序 gather KV，返回 [layers, length, heads, dim]。

        此方法会复制数据，便于学习和调试；paged attention kernel 应直接
        使用 k_cache、v_cache、block_table 和 length，避免每步 gather。
        """
        seq = self.sequences[request_id]
        page_ids = torch.tensor(seq.pages, dtype=torch.long, device=self.k_cache.device)
        k = self.k_cache.index_select(1, page_ids).flatten(1, 2)
        v = self.v_cache.index_select(1, page_ids).flatten(1, 2)
        return k[:, : seq.length], v[:, : seq.length]

    def free(self, request_id: str) -> None:
        """请求结束后回收全部页；旧数据不清零，下次使用前会被覆盖。"""
        seq = self.sequences.pop(request_id)
        self.free_pages.extend(reversed(seq.pages))


if __name__ == "__main__":
    pool = KVPool(
        num_pages=8, page_size=4, num_layers=2, num_kv_heads=2, head_dim=8
    )
    pool.create("req-1")

    # Prefill：6 个 token，占两页，第二页只使用两个槽位。
    k = torch.randn(2, 6, 2, 8, dtype=torch.float16)
    v = torch.randn_like(k)
    pool.append("req-1", k, v)

    # Decode：追加一个 token，继续使用尾页。
    pool.append("req-1", k[:, :1], v[:, :1])
    cached_k, cached_v = pool.read("req-1")
    print("block table:", pool.block_table("req-1"))  # [0, 1]
    print("KV shape:", cached_k.shape)  # [2, 7, 2, 8]
    pool.free("req-1")
    print("free pages:", pool.num_free_pages)  # 8
