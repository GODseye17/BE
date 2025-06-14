"""
Vector Store Management
"""
import logging
from pathlib import Path
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from fastapi import HTTPException

from core.globals import get_globals

logger = logging.getLogger(__name__)

def create_faiss_store_in_batches(docs: List[Document], topic_id: str, batch_size: int = 10):
    """Create FAISS store with batch processing for better performance"""
    vectorstore_path = Path("vectorstores") / str(topic_id)
    vectorstore_path.mkdir(parents=True, exist_ok=True)
    
    if not docs:
        raise ValueError("No documents to process")
    
    globals_dict = get_globals()
    embeddings = globals_dict['embeddings']
    
    logger.info(f"Creating FAISS store with {len(docs)} documents in batches of {batch_size}")
    
    # Process first batch to create the store
    first_batch = docs[:batch_size]
    db = FAISS.from_documents(first_batch, embeddings)
    logger.info(f"Initialized FAISS store with first {len(first_batch)} documents")
    
    # Add remaining documents in batches
    for i in range(batch_size, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        try:
            db.add_documents(batch)
            logger.info(f"Processed batch {i//batch_size + 1} of {(len(docs)-1)//batch_size + 1} ({len(batch)} docs)")
        except Exception as e:
            logger.warning(f"Error processing batch {i//batch_size + 1}: {e}")
            continue
    
    # Save the complete store
    db.save_local(str(vectorstore_path))
    logger.info(f"Saved FAISS store to {vectorstore_path}")
    return db

def get_vectorstore_retriever(topic_id: str, query: str):
    """Get topic-specific FAISS retriever with query-aware k selection"""
    # CRITICAL: Only use topic-specific FAISS, never global stores
    vectorstore_path = Path("vectorstores") / str(topic_id)
    index_path = vectorstore_path / "index.faiss"
    
    # Verify the vector store exists
    if not vectorstore_path.exists():
        logger.error(f"Vectorstore directory not found: {vectorstore_path}")
        raise HTTPException(
            status_code=404, 
            detail=f"No vector store found for topic {topic_id}. Please ensure data fetching completed successfully."
        )
    
    if not index_path.exists():
        logger.error(f"FAISS index file not found: {index_path}")
        raise HTTPException(
            status_code=404, 
            detail=f"FAISS index file not found for topic {topic_id}. The data fetching may have failed."
        )
    
    try:
        globals_dict = get_globals()
        embeddings = globals_dict['embeddings']
        supabase = globals_dict['supabase']
        
        # Load ONLY the topic-specific FAISS store
        logger.info(f"Loading topic-specific FAISS from: {vectorstore_path}")
        db = FAISS.load_local(
            str(vectorstore_path), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Determine k based on query type
        query_lower = query.lower()
        
        # Check if query asks for comprehensive information
        comprehensive_keywords = [
            "all articles", "each article", "every article", 
            "create a table", "list all", "comprehensive", 
            "summary of all", "analyze all", "fetched"
        ]
        
        is_comprehensive = any(keyword in query_lower for keyword in comprehensive_keywords)
        
        if is_comprehensive:
            # For comprehensive queries, get more chunks
            # Get article count from Supabase
            article_count = 20  # default
            if supabase:
                try:
                    result = supabase.table("articles").select("pubmed_id").eq("topic_id", topic_id).execute()
                    article_count = len(result.data) if result.data else 20
                except Exception as e:
                    logger.warning(f"Could not get article count: {e}")
                    article_count = 20
            
            # Use 3 chunks per article for comprehensive queries
            k = min(article_count * 3, 100)
            logger.info(f"📊 Comprehensive query detected - using k={k} for {article_count} articles")
        else:
            # For focused queries, use fewer chunks
            k = 30
            logger.info(f"🎯 Focused query - using k={k}")
        
        # Create retriever with MMR for diversity
        retriever = db.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": k,
                "fetch_k": k * 2,  # Fetch more for MMR to choose from
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )
        
        # Test retrieval
        try:
            test_docs = retriever.get_relevant_documents(query[:100] if len(query) > 100 else query)
            unique_pmids = set(doc.metadata.get('pubmed_id') for doc in test_docs if doc.metadata.get('pubmed_id'))
            logger.info(f"✅ Retrieved {len(test_docs)} chunks from {len(unique_pmids)} unique articles")
            
            # Log if comprehensive query might not have enough articles
            if is_comprehensive and article_count > 0:
                coverage = (len(unique_pmids) / article_count) * 100
                logger.info(f"📈 Article coverage: {len(unique_pmids)}/{article_count} ({coverage:.1f}%)")
                
        except Exception as test_error:
            logger.warning(f"Test retrieval failed: {test_error}, but continuing with retriever")
        
        # Create a custom retriever wrapper that ensures metadata is in content
        return retriever
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error loading topic-specific FAISS: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load vector store: {str(e)}"
        )
