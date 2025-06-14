"""
API Dependencies and background tasks
"""
import asyncio
import logging
from datetime import datetime

from models import TopicRequest
from pubmed.fetcher import fetch_pubmed_data
from core.globals import get_globals
from config.settings import logger

async def fetch_data_background(request: TopicRequest, topic_id: str):
    """Background task to fetch data from PubMed with enhanced multi-topic support"""
    globals_dict = get_globals()
    background_tasks_status = globals_dict['background_tasks_status']
    supabase = globals_dict['supabase']
    
    try:
        background_tasks_status[topic_id] = "processing"
        
        # Convert Pydantic model to function parameters
        filters_dict = None
        if request.filters:
            filters_dict = request.filters.dict(exclude_none=True)
        
        # Set timeout based on number of results requested
        fetch_timeout = 60 + (request.max_results * 2)  # Base 60s + 2s per article
        logger.info(f"Setting fetch timeout to {fetch_timeout}s for {request.max_results} articles")
        try:
            # Run with timeout using the enhanced fetch function
            success = await asyncio.wait_for(
                fetch_pubmed_data(
                    topics=request.topics,
                    operator=request.operator.value if request.operator else "AND",
                    topic=request.topic,  # Backward compatibility
                    topic_id=topic_id,
                    max_results=request.max_results,
                    filters=filters_dict,
                    advanced_query=request.advanced_query
                ),
                timeout=fetch_timeout
            )
            
            if success:
                background_tasks_status[topic_id] = "completed"
            else:
                background_tasks_status[topic_id] = "failed"
        except asyncio.TimeoutError:
            background_tasks_status[topic_id] = "timeout"
            logger.error(f"Fetch operation timed out for topic_id: {topic_id}")
            
            # Update status in Supabase
            if supabase:
                supabase.table("topics").update({"status": "timeout"}).eq("id", topic_id).execute()
                
        except Exception as e:
            background_tasks_status[topic_id] = f"error: {str(e)}"
            logger.error(f"Error in fetch operation for topic_id {topic_id}: {str(e)}")
            
            # Update status in Supabase
            if supabase:
                supabase.table("topics").update({"status": f"error: {str(e)}"}).eq("id", topic_id).execute()
    finally:
        # Keep status for a while but eventually clean up
        await asyncio.sleep(3600)  # Keep status for 1 hour
        if topic_id in background_tasks_status:
            del background_tasks_status[topic_id]