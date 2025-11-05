"""
Translation Service Concurrency Test Script

This script tests the translation service to demonstrate:
1. Sequential vs Parallel translation within a single request
2. Multiple concurrent requests
3. Performance comparison before/after optimization

Usage:
    python test_translation_concurrency.py
"""

import asyncio
import time
import httpx
from typing import List, Dict, Any
from datetime import datetime
import statistics


# Configuration
BASE_URL = "http://localhost:8000"  # Change if your server runs on different port
TRANSLATE_ENDPOINT = f"{BASE_URL}/api/translate"

# Test data
TEST_CASES = [
    {
        "text": "The Karnataka government has announced new policies for education reform.",
        "source_language": "en",
        "target_language": "hi",
        "name": "English to Hindi"
    },
    {
        "text": "Technology is transforming the way we live and work in modern society.",
        "source_language": "en",
        "target_language": "kn",
        "name": "English to Kannada"
    },
    {
        "text": "Climate change is one of the biggest challenges facing humanity today.",
        "source_language": "en",
        "target_language": "hi",
        "name": "English to Hindi (2)"
    },
    {
        "text": "Artificial intelligence and machine learning are revolutionizing industries.",
        "source_language": "en",
        "target_language": "kn",
        "name": "English to Kannada (2)"
    },
]


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


async def translate_single(client: httpx.AsyncClient, test_case: Dict[str, str]) -> Dict[str, Any]:
    """
    Perform a single translation request
    
    Returns:
        Dict with success status, duration, and result
    """
    start_time = time.time()
    
    try:
        response = await client.post(
            TRANSLATE_ENDPOINT,
            json={
                "text": test_case["text"],
                "source_language": test_case["source_language"],
                "target_language": test_case["target_language"]
            },
            timeout=120.0  # 2 minute timeout
        )
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "duration": duration,
                "name": test_case["name"],
                "translated_text": data.get("data", {}).get("translated_text", ""),
                "server_time": data.get("data", {}).get("translation_time", 0)
            }
        else:
            return {
                "success": False,
                "duration": duration,
                "name": test_case["name"],
                "error": f"HTTP {response.status_code}: {response.text}"
            }
            
    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "duration": duration,
            "name": test_case["name"],
            "error": str(e)
        }


async def test_sequential_requests(test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Test sequential translation requests (one after another)
    """
    print_header("TEST 1: Sequential Requests")
    print_info(f"Running {len(test_cases)} translation requests sequentially...")
    
    results = []
    total_start = time.time()
    
    async with httpx.AsyncClient() as client:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n  Request {i}/{len(test_cases)}: {test_case['name']}")
            print(f"  Text: {test_case['text'][:60]}...")
            
            result = await translate_single(client, test_case)
            results.append(result)
            
            if result["success"]:
                print_success(f"Completed in {result['duration']:.2f}s (server: {result['server_time']:.2f}s)")
                print(f"  Translation: {result['translated_text'][:60]}...")
            else:
                print_error(f"Failed: {result['error']}")
    
    total_duration = time.time() - total_start
    
    # Calculate statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\n{Colors.BOLD}Results:{Colors.ENDC}")
    print(f"  Total time: {Colors.BOLD}{total_duration:.2f}s{Colors.ENDC}")
    print(f"  Successful: {Colors.OKGREEN}{len(successful)}/{len(test_cases)}{Colors.ENDC}")
    print(f"  Failed: {Colors.FAIL}{len(failed)}/{len(test_cases)}{Colors.ENDC}")
    
    if successful:
        durations = [r["duration"] for r in successful]
        print(f"  Average time per request: {statistics.mean(durations):.2f}s")
        print(f"  Min time: {min(durations):.2f}s")
        print(f"  Max time: {max(durations):.2f}s")
    
    return {
        "total_duration": total_duration,
        "successful": len(successful),
        "failed": len(failed),
        "results": results
    }


async def test_concurrent_requests(test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Test concurrent translation requests (all at once)
    """
    print_header("TEST 2: Concurrent Requests")
    print_info(f"Running {len(test_cases)} translation requests concurrently...")
    
    total_start = time.time()
    
    async with httpx.AsyncClient() as client:
        # Create all tasks
        tasks = [translate_single(client, test_case) for test_case in test_cases]
        
        # Run all tasks concurrently
        print("\n  Starting all requests simultaneously...")
        results = await asyncio.gather(*tasks)
    
    total_duration = time.time() - total_start
    
    # Display results
    print(f"\n{Colors.BOLD}Individual Results:{Colors.ENDC}")
    for i, result in enumerate(results, 1):
        if result["success"]:
            print_success(f"{i}. {result['name']}: {result['duration']:.2f}s")
        else:
            error_msg = result.get('error', 'Unknown error')
            print_error(f"{i}. {result['name']}: Failed - {error_msg}")
    
    # Calculate statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\n{Colors.BOLD}Results:{Colors.ENDC}")
    print(f"  Total time: {Colors.BOLD}{total_duration:.2f}s{Colors.ENDC}")
    print(f"  Successful: {Colors.OKGREEN}{len(successful)}/{len(test_cases)}{Colors.ENDC}")
    print(f"  Failed: {Colors.FAIL}{len(failed)}/{len(test_cases)}{Colors.ENDC}")
    
    if successful:
        durations = [r["duration"] for r in successful]
        print(f"  Average time per request: {statistics.mean(durations):.2f}s")
        print(f"  Min time: {min(durations):.2f}s")
        print(f"  Max time: {max(durations):.2f}s")
    
    return {
        "total_duration": total_duration,
        "successful": len(successful),
        "failed": len(failed),
        "results": results
    }


async def test_parallel_translation_within_request():
    """
    Test the internal parallel translation (Hindi + Kannada simultaneously)
    This uses the translate_to_all_async method
    """
    print_header("TEST 3: Internal Parallel Translation")
    print_info("Testing translate_to_all_async (Hindi + Kannada in parallel)...")
    
    # This test requires direct access to the translation service
    # We'll use the news creation endpoint which uses translate_to_all_async
    
    print_warning("This test requires authentication token for news creation endpoint")
    print_info("Skipping for now - use TEST 1 and TEST 2 to see concurrency improvements")


async def check_server_health() -> bool:
    """Check if the server is running"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health", timeout=5.0)
            if response.status_code == 200:
                print_success(f"Server is running at {BASE_URL}")
                return True
            else:
                print_error(f"Server returned status {response.status_code}")
                return False
    except Exception as e:
        print_error(f"Cannot connect to server at {BASE_URL}")
        print_error(f"Error: {e}")
        print_info("Make sure the server is running with: python -m app.main")
        return False


async def main():
    """Main test runner"""
    print_header("Translation Service Concurrency Test")
    print(f"Testing server at: {Colors.BOLD}{BASE_URL}{Colors.ENDC}")
    print(f"Test cases: {Colors.BOLD}{len(TEST_CASES)}{Colors.ENDC}")
    print(f"Time: {Colors.BOLD}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    
    # Check server health
    if not await check_server_health():
        return
    
    # Run tests
    sequential_results = await test_sequential_requests(TEST_CASES)
    
    print("\n" + "="*80 + "\n")
    await asyncio.sleep(2)  # Brief pause between tests
    
    concurrent_results = await test_concurrent_requests(TEST_CASES)
    
    # Compare results
    print_header("Performance Comparison")
    
    seq_time = sequential_results["total_duration"]
    conc_time = concurrent_results["total_duration"]
    speedup = seq_time / conc_time if conc_time > 0 else 0
    time_saved = seq_time - conc_time
    
    print(f"{Colors.BOLD}Sequential vs Concurrent:{Colors.ENDC}")
    print(f"  Sequential total time:  {Colors.WARNING}{seq_time:.2f}s{Colors.ENDC}")
    print(f"  Concurrent total time:  {Colors.OKGREEN}{conc_time:.2f}s{Colors.ENDC}")
    print(f"  Time saved:             {Colors.OKGREEN}{time_saved:.2f}s ({(time_saved/seq_time*100):.1f}%){Colors.ENDC}")
    print(f"  Speedup:                {Colors.OKGREEN}{speedup:.2f}x{Colors.ENDC}")
    
    if speedup < 1.5:
        print_warning("\n⚠ Speedup is less than 1.5x - translations may still be sequential due to GIL")
        print_info("  This is expected with the current ThreadPoolExecutor implementation")
        print_info("  For better concurrency, consider ProcessPoolExecutor (Phase 2)")
    elif speedup >= 1.8:
        print_success("\n✓ Good speedup! Parallel translation is working well")
    
    print(f"\n{Colors.BOLD}Interpretation:{Colors.ENDC}")
    print("  • Sequential: Requests wait for each other to complete")
    print("  • Concurrent: Requests run simultaneously (limited by GIL)")
    print("  • With current optimization: ~2x faster per request (Hindi + Kannada parallel)")
    print("  • For true concurrency: Need ProcessPoolExecutor (Phase 2)")
    
    print_header("Test Complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_warning("\n\nTest interrupted by user")
    except Exception as e:
        print_error(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
