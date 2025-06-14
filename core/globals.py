"""
Global instances management
"""
from typing import Optional, Dict, Any
from supabase import Client

# Global instances
_globals = {
    'supabase': None,
    'llm': None,
    'embeddings': None,
    'topic_vectorstores': {},
    'conversation_chains': {},
    'background_tasks_status': {}
}

def set_globals(**kwargs):
    """Set global instances"""
    for key, value in kwargs.items():
        if key in _globals:
            _globals[key] = value
        else:
            raise KeyError(f"Unknown global key: {key}")

def get_globals() -> Dict[str, Any]:
    """Get all global instances"""
    return _globals

def get_global(key: str) -> Any:
    """Get a specific global instance"""
    if key not in _globals:
        raise KeyError(f"Unknown global key: {key}")
    return _globals[key]