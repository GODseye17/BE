"""
API Routes for Vivum RAG Backend
"""
import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks

from models import TopicRequest, QueryRequest, TopicResponse, ChatResponse
from pubmed import QueryPreprocessor
from core.globals import get_globals
from utils import (
    check_topic_fetch_status, get_or_create_chain, 
    validate_comprehensive_response, cleanup_topic_files,
    cleanup_conversation_chains, cleanup_old_topics
)
from pubmed.filters import PubMedFilters
from .dependencies import fetch_data_background

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
def root():
    return {"message": "API is running with multi-topic boolean search support!"}

@router.get("/supabase-status")
async def check_supabase_status():
    globals_dict = get_globals()
    supabase = globals_dict['supabase']
    
    if supabase:
        try:
            # Try a simple query to confirm connection works
            result = supabase.table("topics").select("count").execute()
            return {"status": "connected", "message": "Supabase connection working"}
        except Exception as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}
    else:
        return {"status": "disconnected", "message": "Supabase client not initialized"}

@router.get("/model-status")
async def check_model_status():
    globals_dict = get_globals()
    status = {
        "embedding_model": "loaded" if globals_dict['embeddings'] is not None else "not loaded",
        "llm": "loaded" if globals_dict['llm'] is not None else "not loaded"
    }
    return status

@router.get("/ping")
def ping():
    globals_dict = get_globals()
    background_tasks_status = globals_dict['background_tasks_status']
    return {"status": "alive", "active_tasks": len(background_tasks_status)}
@router.post("/fetch-topic-data", response_model=TopicResponse)
async def fetch_topic_data(request: TopicRequest, background_tasks: BackgroundTasks):
    """Enhanced endpoint to fetch data from PubMed with multi-topic boolean search support"""
    try:
        globals_dict = get_globals()
        supabase = globals_dict['supabase']
        
        # Check if Supabase is connected
        if not supabase:
            raise HTTPException(
                status_code=503,
                detail="Database connection not available"
            )
        
        # Generate a unique topic ID
        topic_id = str(uuid.uuid4())
        
        # Prepare search description for logging
        if request.topics:
            search_description = f"Multi-topic search: {request.topics} with {request.operator}"
        elif request.topic:
            search_description = f"Single topic: {request.topic}"
        elif request.advanced_query:
            search_description = f"Advanced query: {request.advanced_query[:100]}..."
        else:
            search_description = "Unknown search type"
        
        # Create initial record in Supabase with enhanced metadata
        topic_data = {
            "id": topic_id,
            "topic": request.topic,  # Keep for backward compatibility
            "search_topics": ', '.join(request.topics) if request.topics else None,  # Convert list to string
            "boolean_operator": request.operator.value if request.operator else None,
            "advanced_query": request.advanced_query,
            "filters": request.filters.dict(exclude_none=True) if request.filters else None,
            # Remove created_at - it's auto-generated
            "status": "processing"
        }
        
        supabase.table("topics").insert(topic_data).execute()
        
        # Start background task to fetch and store data
        background_tasks.add_task(
            fetch_data_background, 
            request,  # Pass the entire request object
            topic_id
        )
        
        return {
            "topic_id": topic_id,
            "message": f"Started fetching data for: {search_description} (limited to {request.max_results} results)",
            "status": "processing"
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error initiating fetch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=ChatResponse)
async def answer_query(request: QueryRequest):
    """Answer questions using RAG over stored topic articles"""
    try:
        globals_dict = get_globals()
        supabase = globals_dict['supabase']
        llm = globals_dict['llm']
        embeddings = globals_dict['embeddings']
        
        if not supabase:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        status = check_topic_fetch_status(request.topic_id)
        if status != "completed":
            error_map = {
                "processing": (422, "Data is still being fetched. Please try again."),
                "not_found": (404, "No data found. Please fetch the topic data first."),
            }
            code, msg = error_map.get(status, (422, f"Cannot process query. Status: {status}"))
            raise HTTPException(status_code=code, detail=msg)

        if not llm or not embeddings:
            raise HTTPException(status_code=503, detail="LLM or embeddings not loaded.")

        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Set up the chain with dynamic prompt selection
        try:
            chain = get_or_create_chain(request.topic_id, conversation_id, request.query)
            if not chain:
                raise HTTPException(status_code=500, detail="Failed to create conversation chain")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error setting up conversation chain: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error setting up conversational chain: {str(e)}")

        logger.info(f"Processing query: {request.query}")

        try:
            # Invoke chain
            result = chain.invoke({"question": request.query})
            raw_answer = result.get("answer", "Sorry, I couldn't generate an answer to your question.")
            
            # Post-process to remove any system artifacts
            from utils.chains import post_process_response
            answer = post_process_response(raw_answer, request.query)
            
        except Exception as e:
            logger.error(f"Error during chain invocation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing your question: {str(e)}")

        # Validate comprehensive responses
        answer = validate_comprehensive_response(request.query, answer, request.topic_id)

        return {"response": answer, "conversation_id": conversation_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in query processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing your question")
    
    
@router.get("/topic/{topic_id}/status")
async def check_topic_status(topic_id: str):
    """Check the status of data fetching for a topic"""
    status = check_topic_fetch_status(topic_id)
    return {"topic_id": topic_id, "status": status}

@router.get("/topic/{topic_id}/articles")
async def get_topic_articles(topic_id: str, limit: int = 100, offset: int = 0):
    """Fetch all articles for a specific topic"""
    try:
        globals_dict = get_globals()
        supabase = globals_dict['supabase']
        
        # Check if Supabase is connected
        if not supabase:
            raise HTTPException(
                status_code=503,
                detail="Database connection not available"
            )
            
        # First verify the topic exists
        topic_result = supabase.table("topics").select("*").eq("id", topic_id).execute()
        
        if not topic_result.data:
            raise HTTPException(
                status_code=404,
                detail="Topic not found"
            )
            
        # Check if data fetching is complete
        status = check_topic_fetch_status(topic_id)
        if status != "completed":
            return {
                "topic_id": topic_id,
                "status": status,
                "articles": [],
                "message": "Data is still being processed or had an error"
            }
        
        # Fetch articles with pagination
        articles_result = supabase.table("articles") \
            .select("*") \
            .eq("topic_id", topic_id) \
            .range(offset, offset + limit - 1) \
            .execute()
            
        # Get the total count (for pagination info)
        count_result = supabase.table("articles") \
            .select("id", count="exact") \
            .eq("topic_id", topic_id) \
            .execute()
        
        total_count = count_result.count if hasattr(count_result, "count") else len(articles_result.data)
        
        return {
            "topic_id": topic_id,
            "status": "completed",
            "articles": articles_result.data,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count
            }
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error fetching articles for topic {topic_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.delete("/topic/{topic_id}/cleanup")
async def cleanup_topic(topic_id: str):
    """Manually clean up a specific topic's data"""
    try:
        globals_dict = get_globals()
        supabase = globals_dict['supabase']
        
        # Check if topic exists
        if supabase:
            result = supabase.table("topics").select("id, status").eq("id", topic_id).execute()
            if not result.data:
                raise HTTPException(status_code=404, detail="Topic not found")
        
        # Clean up files
        files_cleaned = cleanup_topic_files(topic_id)
        
        # Clean up conversation chains
        chains_cleaned = cleanup_conversation_chains(topic_id)
        
        # Update database status
        if supabase:
            supabase.table("topics").update({
                "status": "cleaned",
                "cleaned_at": datetime.utcnow().isoformat()
            }).eq("id", topic_id).execute()
        
        return {
            "topic_id": topic_id,
            "files_cleaned": files_cleaned,
            "conversation_chains_removed": chains_cleaned,
            "status": "cleaned"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up topic {topic_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup/old-topics")
async def cleanup_old_topics_endpoint(days_old: int = 7):
    """Clean up topics older than specified days"""
    if days_old < 1:
        raise HTTPException(status_code=400, detail="days_old must be at least 1")
    
    result = await cleanup_old_topics(days_old)
    return result

@router.get("/cleanup/status")
async def cleanup_status():
    """Get cleanup system status"""
    import psutil  # You'll need to add 'psutil' to requirements.txt
    
    globals_dict = get_globals()
    conversation_chains = globals_dict['conversation_chains']
    
    # Get disk usage
    vectorstore_dir = Path("vectorstores")
    total_size = 0
    topic_count = 0
    
    if vectorstore_dir.exists():
        for topic_dir in vectorstore_dir.iterdir():
            if topic_dir.is_dir():
                topic_count += 1
                for file in topic_dir.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size
    
    # Get memory usage
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "vector_stores": {
            "count": topic_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "path": str(vectorstore_dir.absolute())
        },
        "conversation_chains": {
            "count": len(conversation_chains),
            "max_allowed": 100
        },
        "memory_usage": {
            "rss_mb": round(memory_info.rss / (1024 * 1024), 2),
            "vms_mb": round(memory_info.vms / (1024 * 1024), 2)
        },
        "cleanup_config": {
            "auto_cleanup_interval_hours": os.getenv("CLEANUP_INTERVAL_HOURS", "24"),
            "auto_cleanup_days_old": os.getenv("CLEANUP_DAYS_OLD", "7")
        }
    }

@router.post("/test-filters")
async def test_filters(request: TopicRequest):
    """Test endpoint to validate filter query construction with multi-topic support"""
    try:
        filter_builder = PubMedFilters()
        
        filters_dict = None
        if request.filters:
            filters_dict = request.filters.dict(exclude_none=True)
        
        final_query = filter_builder.build_complete_query(
            topics=request.topics,
            operator=request.operator.value if request.operator else "AND",
            base_query=request.topic,
            filters=filters_dict,
            advanced_query=request.advanced_query
        )
        
        return {
            "search_method": "multi-topic" if request.topics else "single-topic" if request.topic else "advanced",
            "topics": request.topics,
            "operator": request.operator,
            "original_topic": request.topic,
            "advanced_query": request.advanced_query,
            "filters": filters_dict,
            "final_pubmed_query": final_query
        }
    except Exception as e:
        return {"error": str(e)}
    
@router.get("/test-performance")
async def test_performance():
    """Test endpoint to check system performance"""
    import time
    from langchain.docstore.document import Document
    
    globals_dict = get_globals()
    embeddings = globals_dict['embeddings']
    
    # Test embedding speed
    start = time.time()
    test_docs = [Document(page_content=f"Test document {i}", metadata={"test": i}) for i in range(10)]
    test_embeddings = embeddings.embed_documents([doc.page_content for doc in test_docs])
    embed_time = time.time() - start
    
    return {
        "embedding_model": type(embeddings).__name__,
        "test_embedding_time": f"{embed_time:.2f}s for 10 documents",
        "estimated_time_per_100_docs": f"{embed_time * 10:.1f}s"
    }

@router.post("/transform-query")
async def transform_query(request: dict):
    """Transform natural language query to PubMed syntax"""
    try:
        user_query = request.get("query", "")
        if not user_query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        preprocessor = QueryPreprocessor()
        
        # Check if it's natural language
        is_natural = preprocessor.looks_like_natural_language(user_query)
        
        if not is_natural:
            return {
                "original_query": user_query,
                "transformed_query": user_query,
                "is_transformed": False,
                "explanation": "Query already appears to be in PubMed syntax"
            }
        
        # Transform the query
        transformed = preprocessor.transform_natural_to_pubmed(user_query)
        explanation = preprocessor.get_query_explanation(user_query, transformed)
        
        return {
            "original_query": user_query,
            "transformed_query": transformed,
            "is_transformed": True,
            "explanation": explanation
        }
        
    except Exception as e:
        logger.error(f"Error transforming query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health_check():
    globals_dict = get_globals()
    supabase = globals_dict['supabase']
    return {"status": "healthy", "database": "connected" if supabase else "disconnected"}