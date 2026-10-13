from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Request:
    request_id: int
    token_ids: List[int]
    prompt_length: int = 0
    compute_length: int = 0


@dataclass
class Block:
    block_id: int
    token_ids: List[int]
    hash_value: int
    prev_hash: int
    ref_count: int = 0


@dataclass
class QueryResult:
    matched_blocks: List[Block]
    matched_tokens: int
    remaining_tokens: List[int]


@dataclass
class BlockPool:
    free_block_ids: List[int] = field(default_factory=list)
    used_blocks: Dict[int, Block] = field(default_factory=dict)
    cached_blocks: Dict[int, Block] = field(default_factory=dict)
    next_block_id: int = 0

    def allocate_block(
        self, token_ids: List[int], hash_value: int, prev_hash: int
    ) -> tuple[bool, Block]:
        if hash_value in self.cached_blocks:
            block = self.cached_blocks[hash_value]
            return False, block

        if self.free_block_ids:
            block_id = self.free_block_ids.pop()
        else:
            block_id = self.next_block_id
            self.next_block_id += 1

        block = Block(block_id, token_ids, hash_value, prev_hash)
        self.used_blocks[block_id] = block
        self.cached_blocks[hash_value] = block
        return True, block

    def release_block(self, block_id: int) -> None:
        if block_id in self.used_blocks:
            block = self.used_blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                del self.used_blocks[block_id]
                self.free_block_ids.append(block_id)


class CacheManager:
    def __init__(self):
        self.page_size = 1
        self.request_blocks: Dict[int, List[Block]] = {}
        self.block_pool = BlockPool()

    def _hash_block(self, prev_hash: int, token_ids: List[int]) -> int:
        return hash((prev_hash, tuple(token_ids)))
    
    def query(self, request: Request | List[int]) -> QueryResult:
        token_ids = request.token_ids
        matched_blocks = []
        matched_tokens = 0
        remaining_tokens = token_ids

        prev_hash = 0
        for i in range(0, len(token_ids), self.page_size):
            page_tokens = token_ids[i : i + self.page_size]
            block_hash = self._hash_block(prev_hash, page_tokens)
            block = self.block_pool.cached_blocks.get(block_hash)
            if block is None or block.token_ids != page_tokens:
                break

            if block is not None:
                matched_blocks.append(block)
                matched_tokens += len(page_tokens)
            prev_hash = block_hash
            
        remaining_tokens = token_ids[matched_tokens:]
        return QueryResult(matched_blocks, matched_tokens, remaining_tokens)
    
    def add_request(self, request: Request) -> QueryResult:
        request_blocks = []
        result = self.query(request)

        for block in result.matched_blocks:
            block.ref_count += 1
            request_blocks.append(block)

        prev_hash = request_blocks[-1].hash_value if request_blocks else 0
        for i in range(result.matched_tokens, len(request.token_ids), self.page_size):
            page_tokens = request.token_ids[i : i + self.page_size]
            block_hash = self._hash_block(prev_hash, page_tokens)
            _, block = self.block_pool.allocate_block(page_tokens, block_hash, prev_hash)
            block.ref_count += 1
            request_blocks.append(block)
            prev_hash = block_hash

        request.prompt_length = len(request.token_ids)
        request.compute_length = len(result.remaining_tokens)
        self.request_blocks[request.request_id] = request_blocks
        return result

    def release_request(self, request_id: int) -> None:
        if request_id in self.request_blocks:
            for block in self.request_blocks[request_id]:
                self.block_pool.release_block(block.block_id)
            del self.request_blocks[request_id]

    def add(self, request: Request) -> QueryResult:
        return self.add_request(request)

    def delete(self, request_id: int) -> None:
        self.release_request(request_id)

        


    
