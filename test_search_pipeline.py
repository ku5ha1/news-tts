#!/usr/bin/env python3
"""
Test script for Magazine2 Search Pipeline
This script tests the search pipeline with existing Magazine2 PDFs
"""

import asyncio
import os
import sys
import json
import logging
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_search_pipeline():
    """Test the search pipeline with existing Magazine2 PDFs"""
    try:
        logger.info("Starting Magazine2 Search Pipeline Test")
        
        # Import the pipeline
        from app.services.magazine2_pipeline import Magazine2Pipeline
        
        # Initialize pipeline
        pipeline = Magazine2Pipeline()
        logger.info("Pipeline initialized successfully")
        
        # Test 1: Create search index
        logger.info("Test 1: Creating search index...")
        result = pipeline.create_search_index()
        if result["success"]:
            logger.info("Search index created successfully")
        else:
            logger.error(f"Failed to create search index: {result['error']}")
            return False
        
        # Test 2: Get processing status
        logger.info("Test 2: Getting processing status...")
        status = await pipeline.get_processing_status()
        if status["success"]:
            logger.info(f"Processing status retrieved:")
            logger.info(f"   - Total approved magazines: {status['total_approved']}")
            logger.info(f"   - Processed: {status['processed']}")
            logger.info(f"   - Unprocessed: {status['unprocessed']}")
        else:
            logger.error(f"Failed to get processing status: {status['error']}")
            return False
        
        # Test 3: Process all approved magazines
        if status['unprocessed'] > 0:
            logger.info("Test 3: Processing all approved magazines...")
            result = await pipeline.process_all_approved_magazines()
            if result["success"]:
                logger.info(f"Processing completed:")
                logger.info(f"   - Processed: {result['processed']}")
                logger.info(f"   - Skipped: {result['skipped']}")
                logger.info(f"   - Failed: {result['failed']}")
                logger.info(f"   - Total: {result['total']}")
            else:
                logger.error(f"Failed to process magazines: {result['error']}")
                return False
        else:
            logger.info("Test 3: All magazines already processed, skipping...")
        
        # Test 4: Test search functionality
        logger.info("Test 4: Testing search functionality...")
        from app.services.search_service import SearchService
        search_service = SearchService()
        
        # Test search queries
        test_queries = [
            "budget",
            "Karnataka",
            "government",
            "development",
            "education"
        ]
        
        for query in test_queries:
            logger.info(f"Testing search query: '{query}'")
            results = search_service.search_documents(query, top=3)
            logger.info(f"   Found {len(results)} results")
            for i, result in enumerate(results[:2]):  # Show first 2 results
                logger.info(f"   Result {i+1}: {result['title'][:50]}... (Score: {result['score']:.2f})")
        
        logger.info("Search functionality test completed")
        
        # Test 5: Final status check
        logger.info("Test 5: Final processing status check...")
        final_status = await pipeline.get_processing_status()
        if final_status["success"]:
            logger.info(f"Final status:")
            logger.info(f"   - Total approved magazines: {final_status['total_approved']}")
            logger.info(f"   - Processed: {final_status['processed']}")
            logger.info(f"   - Unprocessed: {final_status['unprocessed']}")
        
        logger.info("All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("Magazine2 Search Pipeline Test")
    logger.info("=" * 60)
    
    # Check environment variables
    required_env_vars = [
        "AZURE_STORAGE_CONNECTION_STRING",
        "AZURE_STORAGE_MAGAZINE2_CONTAINER",
        "AZURE_STORAGE_OUTPUT_CONTAINER_NAME",
        "DOCINT_ENDPOINT",
        "DOCINT_KEY",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_KEY",
        "MONGO_URI"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("All required environment variables are set")
    
    # Run tests
    success = asyncio.run(test_search_pipeline())
    
    if success:
        logger.info("Test completed successfully!")
        return 0
    else:
        logger.error("Test failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
