"""
Global singleton instance for database service to ensure connection reuse.
"""
import logging
from app.services.db_service import DBService

logger = logging.getLogger(__name__)

# Global singleton instance
_db_service_instance = None

def get_db_service() -> DBService:
    """
    Get the singleton DBService instance with connection pooling.
    This ensures all API endpoints reuse the same database connections.
    """
    global _db_service_instance
    if _db_service_instance is None:
        logger.info("[DB_SINGLETON] Creating singleton DBService instance with connection pooling")
        _db_service_instance = DBService()
        logger.info("[DB_SINGLETON] Singleton DBService instance created successfully")
    return _db_service_instance

async def close_db_connections():
    """Close database connections on application shutdown."""
    global _db_service_instance
    if _db_service_instance:
        await _db_service_instance.close_connections()
        _db_service_instance = None
        logger.info("[DB_SINGLETON] Database connections closed")