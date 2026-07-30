import unittest

from kv_cache import CacheCapacityError, KVCacheConfig, KVCacheManager


class KVCacheManagerTest(unittest.TestCase):
    def make_manager(self, *, block_size=4, num_blocks=8, max_context_tokens=32):
        return KVCacheManager(
            KVCacheConfig(
                block_size=block_size,
                num_gpu_blocks=num_blocks,
                max_context_tokens=max_context_tokens,
                model_namespace="demo-model-v1",
            )
        )

    def test_reuses_only_complete_prefix_blocks(self):
        manager = self.make_manager()

        first = manager.create_session("s1", [1, 2, 3, 4, 5, 6])
        second = manager.create_session("s2", [1, 2, 3, 4, 9])

        self.assertEqual(first.reused_tokens, 0)
        self.assertEqual(second.reused_tokens, 4)
        self.assertEqual(
            manager.session_block_ids("s1")[0],
            manager.session_block_ids("s2")[0],
        )
        self.assertNotEqual(
            manager.session_block_ids("s1")[-1],
            manager.session_block_ids("s2")[-1],
        )

    def test_append_uses_copy_on_write_for_a_shared_partial_block(self):
        manager = self.make_manager()
        manager.create_session("parent", [10, 11])
        manager.fork_session("child", "parent")

        parent_block = manager.session_block_ids("parent")[0]
        manager.append_tokens("child", [12])

        self.assertEqual(manager.session_tokens("parent"), (10, 11))
        self.assertEqual(manager.session_tokens("child"), (10, 11, 12))
        self.assertEqual(manager.session_block_ids("parent")[0], parent_block)
        self.assertNotEqual(
            manager.session_block_ids("child")[0],
            parent_block,
        )

    def test_edit_reuses_the_unchanged_full_prefix(self):
        manager = self.make_manager()
        manager.create_session("chat", [1, 2, 3, 4, 5, 6])
        prefix_block = manager.session_block_ids("chat")[0]

        result = manager.replace_session_tokens("chat", [1, 2, 3, 4, 8, 9])

        self.assertEqual(result.reused_tokens, 4)
        self.assertEqual(manager.session_block_ids("chat")[0], prefix_block)
        self.assertEqual(manager.session_tokens("chat"), (1, 2, 3, 4, 8, 9))

    def test_evicts_the_least_recently_used_inactive_prefix(self):
        manager = self.make_manager(block_size=2, num_blocks=2)
        manager.create_session("a", [1, 2])
        manager.release_session("a")
        manager.create_session("b", [3, 4])
        manager.release_session("b")

        manager.create_session("touch-a", [1, 2])
        manager.release_session("touch-a")
        manager.create_session("c", [5, 6])
        manager.release_session("c")

        result = manager.create_session("b-again", [3, 4])
        self.assertEqual(result.reused_tokens, 0)
        self.assertGreaterEqual(manager.metrics.evictions, 1)

    def test_rejects_allocation_when_every_block_is_active(self):
        manager = self.make_manager(block_size=2, num_blocks=1)
        manager.create_session("active", [1, 2])

        with self.assertRaises(CacheCapacityError):
            manager.create_session("other", [3, 4])

    def test_enforces_the_context_window(self):
        manager = self.make_manager(max_context_tokens=4)

        with self.assertRaises(ValueError):
            manager.create_session("too-long", [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
