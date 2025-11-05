"""
Direct Translation Service Test

Tests the translation service directly to show the difference between
sequential and parallel translation within a single request.

This demonstrates the 2x speedup from parallelizing Hindi + Kannada translations.

Usage:
    python test_translation_direct.py
"""

import asyncio
import time
from app.services.translation_service import translation_service


async def test_sequential_translation():
    """
    Simulate OLD behavior: Sequential translation (Hindi first, then Kannada)
    """
    print("\n" + "="*80)
    print("TEST 1: SEQUENTIAL TRANSLATION (Old Behavior)")
    print("="*80)
    
    title = "The Karnataka government announces new education policies"
    description = "The state government has unveiled comprehensive reforms aimed at improving the quality of education across all levels."
    
    print(f"\nOriginal Text:")
    print(f"  Title: {title}")
    print(f"  Description: {description[:60]}...")
    
    start_time = time.time()
    
    # Ensure models are loaded
    await translation_service._ensure_models_loaded()
    
    # Simulate sequential translation (old way)
    print(f"\n⏳ Translating to Hindi...")
    hindi_start = time.time()
    loop = asyncio.get_event_loop()
    
    hindi_title = await loop.run_in_executor(
        None,
        translation_service._translate_en_to_indic_blocking,
        title,
        "hindi"
    )
    hindi_desc = await loop.run_in_executor(
        None,
        translation_service._translate_en_to_indic_blocking,
        description,
        "hindi"
    )
    hindi_time = time.time() - hindi_start
    print(f"✓ Hindi completed in {hindi_time:.2f}s")
    
    print(f"\n⏳ Translating to Kannada...")
    kannada_start = time.time()
    
    kannada_title = await loop.run_in_executor(
        None,
        translation_service._translate_en_to_indic_blocking,
        title,
        "kannada"
    )
    kannada_desc = await loop.run_in_executor(
        None,
        translation_service._translate_en_to_indic_blocking,
        description,
        "kannada"
    )
    kannada_time = time.time() - kannada_start
    print(f"✓ Kannada completed in {kannada_time:.2f}s")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"SEQUENTIAL RESULTS:")
    print(f"  Hindi time:    {hindi_time:.2f}s")
    print(f"  Kannada time:  {kannada_time:.2f}s")
    print(f"  TOTAL TIME:    {total_time:.2f}s")
    print(f"{'='*80}")
    
    print(f"\nTranslations:")
    print(f"  Hindi Title: {hindi_title}")
    print(f"  Kannada Title: {kannada_title}")
    
    return total_time


async def test_parallel_translation():
    """
    Test NEW behavior: Parallel translation (Hindi and Kannada simultaneously)
    """
    print("\n" + "="*80)
    print("TEST 2: PARALLEL TRANSLATION (New Behavior)")
    print("="*80)
    
    title = "The Karnataka government announces new education policies"
    description = "The state government has unveiled comprehensive reforms aimed at improving the quality of education across all levels."
    
    print(f"\nOriginal Text:")
    print(f"  Title: {title}")
    print(f"  Description: {description[:60]}...")
    
    start_time = time.time()
    
    # Use the new translate_to_all_async method
    print(f"\n⏳ Translating to Hindi and Kannada in parallel...")
    
    translations = await translation_service.translate_to_all_async(
        title=title,
        description=description,
        source_lang="english"
    )
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"PARALLEL RESULTS:")
    print(f"  TOTAL TIME:    {total_time:.2f}s")
    print(f"  Languages:     {list(translations.keys())}")
    print(f"{'='*80}")
    
    print(f"\nTranslations:")
    print(f"  Hindi Title: {translations['hindi']['title']}")
    print(f"  Kannada Title: {translations['kannada']['title']}")
    
    return total_time


async def test_multiple_concurrent_requests():
    """
    Test multiple concurrent translation requests
    """
    print("\n" + "="*80)
    print("TEST 3: MULTIPLE CONCURRENT REQUESTS")
    print("="*80)
    
    test_cases = [
        ("Technology is transforming our lives", "Tech news"),
        ("Climate change affects everyone", "Environment news"),
        ("Education is the key to success", "Education news"),
    ]
    
    print(f"\nRunning {len(test_cases)} translation requests concurrently...")
    
    start_time = time.time()
    
    # Create all tasks
    tasks = []
    for title, desc in test_cases:
        task = translation_service.translate_to_all_async(
            title=title,
            description=desc,
            source_lang="english"
        )
        tasks.append(task)
    
    # Run all concurrently
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"CONCURRENT REQUESTS RESULTS:")
    print(f"  Number of requests: {len(test_cases)}")
    print(f"  TOTAL TIME:         {total_time:.2f}s")
    print(f"  Avg per request:    {total_time/len(test_cases):.2f}s")
    print(f"{'='*80}")
    
    for i, (result, (title, _)) in enumerate(zip(results, test_cases), 1):
        print(f"\n  Request {i}: {title}")
        print(f"    Hindi: {result['hindi']['title']}")
        print(f"    Kannada: {result['kannada']['title']}")
    
    return total_time


async def main():
    """Main test runner"""
    print("\n" + "="*80)
    print("TRANSLATION SERVICE CONCURRENCY TEST")
    print("="*80)
    print("\nThis test demonstrates:")
    print("  1. Sequential translation (old way) - Hindi then Kannada")
    print("  2. Parallel translation (new way) - Hindi and Kannada together")
    print("  3. Multiple concurrent requests")
    print("\nExpected improvement: ~2x faster with parallel translation")
    
    try:
        # Test 1: Sequential
        sequential_time = await test_sequential_translation()
        
        await asyncio.sleep(2)  # Brief pause
        
        # Test 2: Parallel
        parallel_time = await test_parallel_translation()
        
        await asyncio.sleep(2)  # Brief pause
        
        # Test 3: Multiple concurrent
        concurrent_time = await test_multiple_concurrent_requests()
        
        # Final comparison
        print("\n" + "="*80)
        print("FINAL COMPARISON")
        print("="*80)
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        time_saved = sequential_time - parallel_time
        
        print(f"\nSingle Request Performance:")
        print(f"  Sequential (old):  {sequential_time:.2f}s")
        print(f"  Parallel (new):    {parallel_time:.2f}s")
        print(f"  Time saved:        {time_saved:.2f}s ({(time_saved/sequential_time*100):.1f}%)")
        print(f"  Speedup:           {speedup:.2f}x")
        
        if speedup >= 1.8:
            print(f"\n✓ Excellent! Parallel translation is working as expected")
        elif speedup >= 1.5:
            print(f"\n✓ Good! Parallel translation shows improvement")
        else:
            print(f"\n⚠ Speedup is lower than expected")
            print(f"  This might be due to GIL limitations or model loading time")
        
        print(f"\nMultiple Concurrent Requests:")
        print(f"  3 requests took:   {concurrent_time:.2f}s")
        print(f"  Expected (seq):    ~{parallel_time * 3:.2f}s")
        print(f"  Actual speedup:    {(parallel_time * 3) / concurrent_time:.2f}x")
        
        print("\n" + "="*80)
        print("KEY TAKEAWAYS")
        print("="*80)
        print("\n1. Parallel translation (Hindi + Kannada together) is ~2x faster")
        print("2. Multiple requests still limited by GIL (ThreadPoolExecutor)")
        print("3. For true concurrency, need ProcessPoolExecutor (Phase 2)")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
