"""
Main entry point for Vivum RAG Backend
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from supabase import create_client

from api import router
from config.settings import (
    SUPABASE_URL, SUPABASE_KEY, TOGETHER_API_KEY, LLM_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, EMBEDDING_MODEL,
    CLEANUP_INTERVAL_HOURS, CLEANUP_DAYS_OLD, PORT
)
from core import set_globals
from llm import TogetherChatModel
from utils import cleanup_old_topics

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager"""
    # Startup: Initialize connections and models
    logger.info("Starting application: Initializing connections and models")
    
    try:
        # Initialize Supabase
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase connection established")
        
        # Initialize embedding model
        logger.info("Loading embedding model")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
        
        # Initialize LLM
        logger.info("Loading Llama model")
        try:
            llm = TogetherChatModel(
                api_key=TOGETHER_API_KEY,
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                streaming=True
            )
            logger.info("LLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            llm = None
        
        # Set global instances
        set_globals(
            supabase=supabase,
            llm=llm,
            embeddings=embeddings,
            topic_vectorstores={},
            conversation_chains={},
            background_tasks_status={}
        )
        
        # Start automatic cleanup task
        async def auto_cleanup_task():
            """Automatically clean up old topics periodically"""
            while True:
                try:
                    await asyncio.sleep(CLEANUP_INTERVAL_HOURS * 3600)  # Convert hours to seconds
                    logger.info(f"🧹 Running automatic cleanup for topics older than {CLEANUP_DAYS_OLD} days")
                    result = await cleanup_old_topics(CLEANUP_DAYS_OLD)
                    logger.info(f"🧹 Automatic cleanup completed: {result}")
                except Exception as e:
                    logger.error(f"Error in automatic cleanup: {e}")
        
        # Create the cleanup task
        asyncio.create_task(auto_cleanup_task())
        logger.info(f"🧹 Automatic cleanup scheduled every {CLEANUP_INTERVAL_HOURS} hours for topics older than {CLEANUP_DAYS_OLD} days")
        
        logger.info("Using topic-specific FAISS stores only - no global vector stores")
            
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
    
    yield
    
    # Shutdown: Clean up resources
    logger.info("Application shutdown: Cleaning up resources")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Vivum RAG Backend",
    description="A production-ready Retrieval-Augmented Generation (RAG) system for researchers to query PubMed articles with AI-powered insights.",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.vivum.app", "http://localhost:8081", "http://localhost:3000", 'https://frontend-vivum.vercel.app' , 'https://www.pubmed.app'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

if __name__ == "__main__":
    # Use uvicorn to run the app
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=PORT,
        workers=1,  # Single worker to avoid memory issues
        log_level="info",
        timeout_keep_alive=65  # Railway closes idle connections after 75s
    )