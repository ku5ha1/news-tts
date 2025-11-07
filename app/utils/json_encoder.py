"""Universal JSON encoder for MongoDB documents."""
from bson import ObjectId
from datetime import datetime
from typing import Any


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert MongoDB documents to JSON-serializable format.
    Handles ObjectIds, datetime objects, and nested structures.
    """
    if isinstance(obj, ObjectId):
        return {"$oid": str(obj)}
    
    elif isinstance(obj, datetime):
        return {"$date": obj.replace(microsecond=0).isoformat() + "Z"}
    
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    
    else:
        # Return as-is for primitives (str, int, float, bool, None)
        return obj


def to_extended_json(document: dict) -> dict:
    """
    Convert a MongoDB document to extended JSON format.
    This is a convenience wrapper around convert_to_json_serializable.
    """
    return convert_to_json_serializable(document)

