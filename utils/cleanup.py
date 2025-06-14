"""
Cleanup utilities for managing old topics and resources
"""
import logging
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict

from core.globals import get_globals

logger = logging.getLogger(__name__)

def cleanup_topic_files(topic_id: str) -> bool:
    """Clean up FAISS files for a specific topic"""
    try:
        vectorstore_path = Path("vectorstores") / str(topic_id)
        
        if vectorstore_path.exists():
            shutil.rmtree(vectorstore_path)
            logger.info(f"🗑️ Cleaned up vector store files for topic {topic_id}")
            return True
        else:
            logger.warning(f"Vector store path not found for topic {topic_id}")
            return False
    except Exception as e:
        logger.error(f"Error cleaning up topic {topic_id}: {e}")
        return False

def cleanup_conversation_chains(topic_id: str) -> int:
    """Clean up conversation chains for a specific topic"""
    globals_dict = get_globals()
    conversation_chains = globals_dict['conversation_chains']
    
    chains_removed = 0
    keys_to_remove = []
    
    for key in conversation_chains.keys():
        if key.startswith(f"{topic_id}:"):
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del conversation_chains[key]
        chains_removed += 1
    
    logger.info(f"🗑️ Removed {chains_removed} conversation chains for topic {topic_id}")
    return chains_removed

async def cleanup_old_topics(days_old: int = 7) -> Dict[str, int]:
    """Clean up topics older than specified days"""
    globals_dict = get_globals()
    supabase = globals_dict['supabase']
    
    if not supabase:
        return {"error": "Database not connected"}
    
    try:
        # Calculate cutoff date
        cutoff_date = (datetime.utcnow() - timedelta(days=days_old)).isoformat()
        
        # Get old topics
        result = supabase.table("topics") \
            .select("id, created_at") \
            .lt("created_at", cutoff_date) \
            .execute()
        
        old_topics = result.data if result.data else []
        
        cleaned_count = 0
        failed_count = 0
        
        for topic in old_topics:
            topic_id = topic["id"]
            try:
                # Clean up files
                file_cleaned = cleanup_topic_files(topic_id)
                
                # Clean up conversation chains
                chains_cleaned = cleanup_conversation_chains(topic_id)
                
                # Mark as cleaned in database (optional - you could also delete the record)
                supabase.table("topics").update({
                    "status": "cleaned",
                    "cleaned_at": datetime.utcnow().isoformat()
                }).eq("id", topic_id).execute()
                
                if file_cleaned:
                    cleaned_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to clean topic {topic_id}: {e}")
                failed_count += 1
        
        return {
            "total_old_topics": len(old_topics),
            "cleaned": cleaned_count,
            "failed": failed_count,
            "cutoff_date": cutoff_date
        }
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_topics: {e}")
        return {"error": str(e)}