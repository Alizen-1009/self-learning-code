import unittest

from python.LLM.perfix_cache import CacheManager, Request


class CacheManagerTests(unittest.TestCase):
    def test_query_returns_longest_cached_block_prefix(self) -> None:
        manager = CacheManager()
        manager.add(Request(request_id=1, token_ids=[10, 20, 30, 40]))

        result = manager.query([10, 20, 30, 99])

        self.assertEqual(result.matched_tokens, 3)
        self.assertEqual(result.remaining_tokens, [99])
        self.assertEqual(
            [block.token_ids for block in result.matched_blocks],
            [[10], [20], [30]],
        )

    def test_add_sets_compute_length_from_cache_hit(self) -> None:
        manager = CacheManager()
        manager.add(Request(request_id=1, token_ids=[10, 20, 30, 40]))

        request = Request(request_id=2, token_ids=[10, 20, 30, 99])
        manager.add(request)

        self.assertEqual(request.prompt_length, 4)
        self.assertEqual(request.compute_length, 1)
        self.assertIs(manager.request_blocks[1][0], manager.request_blocks[2][0])

    def test_delete_releases_request_but_keeps_prefix_cache(self) -> None:
        manager = CacheManager()
        request = Request(request_id=1, token_ids=[1, 2, 3, 4])
        manager.add(request)

        first_block = manager.request_blocks[1][0]
        manager.delete(1)
        result = manager.query([1, 2, 9, 9])

        self.assertEqual(first_block.ref_count, 0)
        self.assertNotIn(1, manager.request_blocks)
        self.assertEqual(result.matched_tokens, 2)


if __name__ == "__main__":
    unittest.main()
