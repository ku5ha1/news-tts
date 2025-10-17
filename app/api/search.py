import time
import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from app.models.search import (
    SearchRequest, SearchResponse, SearchResult,
    ProcessingStatusRequest, ProcessingStatusResponse,
    ProcessMagazineRequest, ProcessMagazineResponse,
    ProcessAllRequest, ProcessAllResponse,
    CreateIndexRequest, CreateIndexResponse,
    ErrorResponse
)
from app.services.search_service import SearchService
from app.services.magazine2_pipeline import Magazine2Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])

def get_search_service() -> SearchService:
    """Dependency to get search service instance"""
    return SearchService()

def get_magazine2_pipeline() -> Magazine2Pipeline:
    """Dependency to get magazine2 pipeline instance"""
    return Magazine2Pipeline()

@router.post("/query", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service)
):
    """
    Search documents using semantic and vector search
    """
    try:
        start_time = time.time()
        
        logger.info(f"Search request: {request.query}")
        
        # Perform search
        search_results = search_service.search_documents(
            query=request.query,
            top=request.top
        )
        
        # Convert to response format
        results = []
        for result in search_results:
            search_result = SearchResult(
                id=result["id"],
                title=result["title"],
                description=result["description"],
                content=result["content"],
                magazine_id=result["magazine_id"],
                published_year=result["published_year"],
                published_month=result["published_month"],
                edition_number=result["edition_number"],
                pdf_url=result["pdf_url"],
                thumbnail_url=result["thumbnail_url"],
                score=result["score"]
            )
            results.append(search_result)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        response = SearchResponse(
            success=True,
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time_ms=processing_time
        )
        
        logger.info(f"Search completed: {len(results)} results in {processing_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/status", response_model=ProcessingStatusResponse)
async def get_processing_status(
    pipeline: Magazine2Pipeline = Depends(get_magazine2_pipeline)
):
    """
    Get the current processing status of magazines
    """
    try:
        logger.info(f"Processing status request")
        
        status = await pipeline.get_processing_status()
        
        if not status["success"]:
            raise HTTPException(status_code=500, detail=status["error"])
        
        response = ProcessingStatusResponse(
            success=True,
            total_approved=status["total_approved"],
            processed=status["processed"],
            unprocessed=status["unprocessed"],
            processed_files=status["processed_files"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get processing status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get processing status: {str(e)}")

@router.post("/process/single", response_model=ProcessMagazineResponse)
async def process_single_magazine(
    request: ProcessMagazineRequest,
    pipeline: Magazine2Pipeline = Depends(get_magazine2_pipeline)
):
    """
    Process a single magazine through the search pipeline
    """
    try:
        logger.info(f"Single magazine processing request: {request.magazine_id}")
        
        result = await pipeline.process_single_magazine(request.magazine_id)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        response = ProcessMagazineResponse(
            success=True,
            message=result["message"],
            skipped=result.get("skipped")
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process single magazine: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process magazine: {str(e)}")

@router.post("/process/all", response_model=ProcessAllResponse)
async def process_all_magazines(
    pipeline: Magazine2Pipeline = Depends(get_magazine2_pipeline)
):
    """
    Process all approved magazines through the search pipeline
    """
    try:
        logger.info(f"Process all magazines request")
        
        result = await pipeline.process_all_approved_magazines()
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        response = ProcessAllResponse(
            success=True,
            processed=result["processed"],
            skipped=result["skipped"],
            failed=result["failed"],
            total=result["total"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process all magazines: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process magazines: {str(e)}")

@router.post("/index/create", response_model=CreateIndexResponse)
async def create_search_index(
    pipeline: Magazine2Pipeline = Depends(get_magazine2_pipeline)
):
    """
    Create or update the search index
    """
    try:
        logger.info(f"Create search index request")
        
        result = pipeline.create_search_index()
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        response = CreateIndexResponse(
            success=True,
            message=result["message"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create search index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create search index: {str(e)}")

@router.get("/health")
async def search_health_check():
    """
    Health check endpoint for search service
    """
    try:
        search_service = SearchService()
        # Try to create search index to verify all services are working
        success = search_service.create_search_index()
        
        if success:
            return {"status": "healthy", "message": "Search service is operational"}
        else:
            return {"status": "unhealthy", "message": "Search service has issues"}
            
    except Exception as e:
        logger.error(f"Search health check failed: {e}")
        return {"status": "unhealthy", "message": f"Search service error: {str(e)}"}
