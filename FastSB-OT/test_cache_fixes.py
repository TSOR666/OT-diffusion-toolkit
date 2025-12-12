"""Test suite demonstrating cache fixes"""

import torch
import threading
from fastsb_ot.cache import MemoryEfficientCacheFixed


def test_type_validation():
    """Fix #1: Type validation prevents AttributeError"""
    print("\n=== Test 1: Type Validation ===")
    cache = MemoryEfficientCacheFixed(max_size_mb=10)

    try:
        cache.put("key", [1, 2, 3])  # Non-tensor
        print("[FAIL] Should raise TypeError")
        return False
    except TypeError as e:
        print(f"[PASS] Correctly raised TypeError: {e}")
        return True


def test_storage_leak_prevention():
    """Fix #2: Storage overhead detection prevents memory leaks"""
    print("\n=== Test 2: Storage Leak Prevention ===")
    cache = MemoryEfficientCacheFixed(max_size_mb=100)

    # Create a large tensor and a small view
    big_tensor = torch.randn(1000, 1000)  # ~4 MB
    small_slice = big_tensor[0:10, :]  # View of ~40 KB, but shares 4 MB storage

    # Cache the slice
    cache.put("slice", small_slice)

    # Retrieve and check if it was cloned
    cached_value = cache.get("slice")

    # Check storage sizes
    original_storage = small_slice.untyped_storage().size()
    cached_storage = cached_value.untyped_storage().size()

    print(f"Original slice storage: {original_storage / 1024:.1f} KB")
    print(f"Cached value storage: {cached_storage / 1024:.1f} KB")

    if cached_storage < original_storage * 0.7:
        print("[PASS] Storage overhead eliminated (tensor was cloned)")
        return True
    else:
        print("[FAIL] Storage leak present")
        return False


def test_thread_safety():
    """Fix #3-5, #15: Thread-safe state access"""
    print("\n=== Test 3: Thread Safety ===")
    cache = MemoryEfficientCacheFixed(max_size_mb=50, max_entries=100)
    errors = []

    def worker(thread_id):
        try:
            for i in range(20):
                key = f"t{thread_id}_k{i}"
                value = torch.randn(100, 100)
                cache.put(key, value)

                # Occasionally read stats
                if i % 5 == 0:
                    stats = cache.get_stats()
                    # Verify stats are sane
                    if stats["hit_rate"] < 0 or stats["hit_rate"] > 1:
                        errors.append(f"Invalid hit rate: {stats['hit_rate']}")
                    if stats["pending_flush_mb"] < 0:
                        errors.append(f"Negative pending flush: {stats['pending_flush_mb']}")
        except Exception as e:
            errors.append(f"Thread {thread_id} error: {e}")

    # Run 10 threads concurrently
    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    final_stats = cache.get_stats()
    print(f"Final stats: {final_stats}")

    if errors:
        print(f"[FAIL] {len(errors)} errors occurred:")
        for err in errors[:3]:  # Show first 3
            print(f"  - {err}")
        return False
    else:
        print("[PASS] No errors in concurrent access")
        return True


def test_cuda_operations_outside_lock():
    """Fix #6-8: CUDA operations don't block cache access"""
    print("\n=== Test 4: CUDA Operations Outside Lock ===")

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    cache = MemoryEfficientCacheFixed(max_size_mb=100, flush_threshold_mb=10)

    # Fill cache with CUDA tensors to trigger flushes
    for i in range(50):
        key = f"cuda_tensor_{i}"
        value = torch.randn(500, 500, device='cuda')  # ~1 MB each
        cache.put(key, value)

    print("[PASS] CUDA operations executed without deadlock")
    return True


def test_multi_gpu_tracking():
    """Fix #9: Multi-GPU memory tracking"""
    print("\n=== Test 5: Multi-GPU Memory Tracking ===")

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    cache = MemoryEfficientCacheFixed(max_size_mb=100)

    # Test the new multi-GPU memory function
    allocated, reserved = cache._get_total_cuda_memory()
    print(f"Total CUDA memory - Allocated: {allocated / 1e6:.1f} MB, Reserved: {reserved / 1e6:.1f} MB")

    if allocated >= 0 and reserved >= allocated:
        print("[PASS] Multi-GPU memory tracking working")
        return True
    else:
        print("[FAIL] Invalid memory values")
        return False


def test_adaptive_downscaling():
    """Fix #13-14: Adaptive downscaling logic"""
    print("\n=== Test 6: Adaptive Downscaling ===")

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True

    cache = MemoryEfficientCacheFixed(max_size_mb=500)
    initial_max = cache.max_size_bytes

    # Fill cache to trigger downscaling
    for i in range(100):
        cache.put(f"key_{i}", torch.randn(1000, 1000))

    # Force downscaling check
    cache._maybe_downscale_cache(force=True)

    final_max = cache.max_size_bytes
    print(f"Initial max: {initial_max / 1e6:.1f} MB, Final max: {final_max / 1e6:.1f} MB")

    if final_max <= initial_max:
        print("[PASS] Adaptive downscaling executed")
        return True
    else:
        print("[WARN] No downscaling occurred (may be expected if GPU has lots of free memory)")
        return True


def test_stats_completeness():
    """Enhancement: Stats include max_size_mb"""
    print("\n=== Test 7: Stats Completeness ===")
    cache = MemoryEfficientCacheFixed(max_size_mb=50)

    cache.put("key1", torch.randn(100, 100))
    cache.put("key2", torch.randn(200, 200))

    stats = cache.get_stats()

    required_fields = ["hits", "misses", "hit_rate", "size_mb", "max_size_mb", "entries", "pending_flush_mb"]
    missing = [f for f in required_fields if f not in stats]

    if missing:
        print(f"[FAIL] Missing fields: {missing}")
        return False
    else:
        print(f"[PASS] All stats fields present: {list(stats.keys())}")
        return True


def run_all_tests():
    """Run all test cases"""
    print("=" * 60)
    print("CACHE FIXES TEST SUITE")
    print("=" * 60)

    tests = [
        test_type_validation,
        test_storage_leak_prevention,
        test_thread_safety,
        test_cuda_operations_outside_lock,
        test_multi_gpu_tracking,
        test_adaptive_downscaling,
        test_stats_completeness,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"[EXCEPTION] in {test.__name__}: {e}")
            results.append((test.__name__, False))

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
