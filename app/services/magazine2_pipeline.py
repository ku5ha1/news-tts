import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.search_service import SearchService
from app.services.db_service import DBService
from bson import ObjectId

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Magazine2Pipeline:
    """Pipeline for processing Magazine2 PDFs through the search system"""
    
    def __init__(self):
        """Initialize the pipeline"""
        self.search_service = SearchService()
        self.db_service = DBService()
        self.db = self.db_service.db
        self.magazine2_collection = self.db["magazine2"]
    
    async def process_all_approved_magazines(self) -> Dict[str, Any]:
        """Process all approved Magazine2 documents"""
        try:
            logger.info("Starting processing of all approved Magazine2 documents")
            
            # Get all approved magazines
            cursor = self.magazine2_collection.find({"status": "approved"})
            approved_magazines = await cursor.to_list(length=None)
            
            if not approved_magazines:
                logger.info("No approved magazines found")
                return {"success": True, "processed": 0, "skipped": 0, "failed": 0}
            
            logger.info(f"Found {len(approved_magazines)} approved magazines")
            
            processed_count = 0
            skipped_count = 0
            failed_count = 0
            
            for magazine in approved_magazines:
                magazine_id = str(magazine["_id"])
                
                # Check if already processed
                if self.search_service.is_file_processed(magazine_id):
                    logger.info(f"Magazine {magazine_id} already processed, skipping")
                    skipped_count += 1
                    continue
                
                # Process the magazine
                if self.search_service.process_magazine2_pdf(magazine):
                    processed_count += 1
                    logger.info(f"Successfully processed magazine {magazine_id}")
                else:
                    failed_count += 1
                    logger.error(f"Failed to process magazine {magazine_id}")
            
            result = {
                "success": True,
                "processed": processed_count,
                "skipped": skipped_count,
                "failed": failed_count,
                "total": len(approved_magazines)
            }
            
            logger.info(f"Processing complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process magazines: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_single_magazine(self, magazine_id: str) -> Dict[str, Any]:
        """Process a single Magazine2 document"""
        try:
            logger.info(f"Processing single magazine: {magazine_id}")
            
            # Get magazine from database
            magazine = await self.magazine2_collection.find_one({"_id": ObjectId(magazine_id)})
            
            if not magazine:
                return {"success": False, "error": "Magazine not found"}
            
            if magazine.get("status") != "approved":
                return {"success": False, "error": f"Magazine status is '{magazine.get('status')}', not 'approved'"}
            
            # Check if already processed
            if self.search_service.is_file_processed(magazine_id):
                return {"success": True, "message": "Magazine already processed", "skipped": True}
            
            # Process the magazine
            if self.search_service.process_magazine2_pdf(magazine):
                return {"success": True, "message": "Magazine processed successfully"}
            else:
                return {"success": False, "error": "Failed to process magazine"}
                
        except Exception as e:
            logger.error(f"Failed to process magazine {magazine_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_new_magazines(self, since_timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Process magazines created since a specific timestamp"""
        try:
            logger.info("Processing new magazines")
            
            # Default to last 24 hours if no timestamp provided
            if not since_timestamp:
                since_timestamp = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Get magazines created since timestamp
            query = {
                "status": "approved",
                "createdTime": {"$gte": since_timestamp}
            }
            
            cursor = self.magazine2_collection.find(query)
            new_magazines = await cursor.to_list(length=None)
            
            if not new_magazines:
                logger.info("No new magazines found")
                return {"success": True, "processed": 0, "skipped": 0, "failed": 0}
            
            logger.info(f"Found {len(new_magazines)} new magazines")
            
            processed_count = 0
            skipped_count = 0
            failed_count = 0
            
            for magazine in new_magazines:
                magazine_id = str(magazine["_id"])
                
                # Check if already processed
                if self.search_service.is_file_processed(magazine_id):
                    logger.info(f"Magazine {magazine_id} already processed, skipping")
                    skipped_count += 1
                    continue
                
                # Process the magazine
                if self.search_service.process_magazine2_pdf(magazine):
                    processed_count += 1
                    logger.info(f"Successfully processed magazine {magazine_id}")
                else:
                    failed_count += 1
                    logger.error(f"Failed to process magazine {magazine_id}")
            
            result = {
                "success": True,
                "processed": processed_count,
                "skipped": skipped_count,
                "failed": failed_count,
                "total": len(new_magazines)
            }
            
            logger.info(f"New magazines processing complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process new magazines: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_processing_status(self) -> Dict[str, Any]:
        """Get the current processing status"""
        try:
            # Get all approved magazines
            cursor = self.magazine2_collection.find({"status": "approved"})
            approved_magazines = await cursor.to_list(length=None)
            
            # Get processed files
            processed_files = self.search_service.get_processed_files()
            
            # Count processed vs unprocessed
            processed_count = 0
            unprocessed_count = 0
            
            for magazine in approved_magazines:
                magazine_id = str(magazine["_id"])
                if magazine_id in processed_files:
                    processed_count += 1
                else:
                    unprocessed_count += 1
            
            return {
                "success": True,
                "total_approved": len(approved_magazines),
                "processed": processed_count,
                "unprocessed": unprocessed_count,
                "processed_files": processed_files
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing status: {e}")
            return {"success": False, "error": str(e)}
    
    def create_search_index(self) -> Dict[str, Any]:
        """Create the search index"""
        try:
            success = self.search_service.create_search_index()
            if success:
                return {"success": True, "message": "Search index created successfully"}
            else:
                return {"success": False, "error": "Failed to create search index"}
        except Exception as e:
            logger.error(f"Failed to create search index: {e}")
            return {"success": False, "error": str(e)}

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Magazine2 Search Pipeline")
    parser.add_argument("--action", choices=["process_all", "process_single", "process_new", "status", "create_index"], 
                       required=True, help="Action to perform")
    parser.add_argument("--magazine_id", help="Magazine ID for single processing")
    parser.add_argument("--since", help="Timestamp for new processing (ISO format)")
    
    args = parser.parse_args()
    
    pipeline = Magazine2Pipeline()
    
    if args.action == "process_all":
        result = pipeline.process_all_approved_magazines()
    elif args.action == "process_single":
        if not args.magazine_id:
            print("Error: --magazine_id required for process_single")
            return
        result = pipeline.process_single_magazine(args.magazine_id)
    elif args.action == "process_new":
        since_timestamp = None
        if args.since:
            since_timestamp = datetime.fromisoformat(args.since.replace('Z', '+00:00'))
        result = pipeline.process_new_magazines(since_timestamp)
    elif args.action == "status":
        result = pipeline.get_processing_status()
    elif args.action == "create_index":
        result = pipeline.create_search_index()
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
