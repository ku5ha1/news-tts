from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class SearchRequest(BaseModel):
    """Request model for search functionality"""
    query: str = Field(..., description="Search query text", min_length=1, max_length=500)
    top: Optional[int] = Field(default=10, description="Number of results to return", ge=1, le=50)
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters for search")
    vector_weight: Optional[float] = Field(default=None, description="Weight for vector component in hybrid search (e.g., 2.5-4.0)")

class SearchResult(BaseModel):
    """Individual search result model"""
    id: str = Field(..., description="Unique identifier for the search result")
    title: str = Field(..., description="Title of the magazine")
    description: str = Field(..., description="Description of the magazine")
    content: str = Field(..., description="Relevant content snippet")
    magazine_id: str = Field(..., description="ID of the source magazine")
    published_year: int = Field(..., description="Year of publication")
    published_month: str = Field(..., description="Month of publication")
    edition_number: str = Field(..., description="Edition number")
    pdf_url: str = Field(..., description="URL to the PDF file")
    thumbnail_url: str = Field(..., description="URL to the thumbnail image")
    score: float = Field(..., description="Relevance score")

class SearchResponse(BaseModel):
    """Response model for search functionality"""
    success: bool = Field(..., description="Whether the search was successful")
    query: str = Field(..., description="The original search query")
    results: List[SearchResult] = Field(..., description="List of search results")
    total_results: int = Field(..., description="Total number of results found")
    processing_time_ms: Optional[float] = Field(default=None, description="Time taken to process the search")

class ProcessingStatusRequest(BaseModel):
    """Request model for getting processing status"""
    pass

class ProcessingStatusResponse(BaseModel):
    """Response model for processing status"""
    success: bool = Field(..., description="Whether the request was successful")
    total_approved: int = Field(..., description="Total number of approved magazines")
    processed: int = Field(..., description="Number of processed magazines")
    unprocessed: int = Field(..., description="Number of unprocessed magazines")
    processed_files: List[str] = Field(..., description="List of processed file IDs")

class ProcessMagazineRequest(BaseModel):
    """Request model for processing a single magazine"""
    magazine_id: str = Field(..., description="ID of the magazine to process")

class ProcessMagazineResponse(BaseModel):
    """Response model for processing a single magazine"""
    success: bool = Field(..., description="Whether the processing was successful")
    message: str = Field(..., description="Status message")
    skipped: Optional[bool] = Field(default=None, description="Whether the magazine was skipped")

class ProcessAllRequest(BaseModel):
    """Request model for processing all approved magazines"""
    pass

class ProcessAllResponse(BaseModel):
    """Response model for processing all approved magazines"""
    success: bool = Field(..., description="Whether the processing was successful")
    processed: int = Field(..., description="Number of magazines processed")
    skipped: int = Field(..., description="Number of magazines skipped")
    failed: int = Field(..., description="Number of magazines that failed to process")
    total: int = Field(..., description="Total number of magazines")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")

class CreateIndexRequest(BaseModel):
    """Request model for creating search index"""
    pass

class CreateIndexResponse(BaseModel):
    """Response model for creating search index"""
    success: bool = Field(..., description="Whether the index creation was successful")
    message: str = Field(..., description="Status message")
    error: Optional[str] = Field(default=None, description="Error message if creation failed")

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(default=False, description="Always false for error responses")
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(default=None, description="Additional error details")
