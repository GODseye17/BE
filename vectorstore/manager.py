"""
Vector Store Management with Performance Optimizations
"""
import logging
from pathlib import Path
from typing import List, Optional
import concurrent.futures
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from fastapi import HTTPException

from core.globals import get_globals

logger = logging.getLogger(__name__)

def create_faiss_store_in_batches(docs: List[Document], topic_id: str, batch_size: int = 50):
    """Create FAISS store with parallel batch processing for better performance"""
    vectorstore_path = Path("vectorstores") / str(topic_id)
    vectorstore_path.mkdir(parents=True, exist_ok=True)
    
    if not docs:
        raise ValueError("No documents to process")
    
    globals_dict = get_globals()
    embeddings = globals_dict['embeddings']
    
    logger.info(f"Creating FAISS store with {len(docs)} documents in batches of {batch_size}")
    
    # Parallel embedding generation for faster processing
    all_embeddings = []
    all_docs = []
    
    # Use ThreadPoolExecutor for parallel embedding generation
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all embedding tasks
        future_to_docs = {}
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            # Extract texts for embedding
            texts = [doc.page_content for doc in batch]
            # Submit embedding task
            future = executor.submit(embeddings.embed_documents, texts)
            future_to_docs[future] = batch
            logger.info(f"Submitted batch {i//batch_size + 1} of {(len(docs)-1)//batch_size + 1} for embedding")
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_docs):
            batch_docs = future_to_docs[future]
            try:
                batch_embeddings = future.result()
                all_embeddings.extend(batch_embeddings)
                all_docs.extend(batch_docs)
                logger.info(f"Completed embedding batch with {len(batch_docs)} documents")
            except Exception as e:
                logger.warning(f"Error processing batch: {e}")
                # Still add docs even if embedding fails
                # Use zero vectors as fallback
                fallback_embeddings = [[0.0] * 768] * len(batch_docs)  # Assuming 768-dim embeddings
                all_embeddings.extend(fallback_embeddings)
                all_docs.extend(batch_docs)
    
    # Create FAISS index from all embeddings at once (faster than incremental)
    logger.info(f"Creating FAISS index with {len(all_embeddings)} embeddings")
    
    if all_embeddings and all_docs:
        # Create the vector store with all embeddings at once
        texts = [doc.page_content for doc in all_docs]
        metadatas = [doc.metadata for doc in all_docs]
        
        db = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, all_embeddings)),
            embedding=embeddings,
            metadatas=metadatas
        )
        
        # Save the complete store
        db.save_local(str(vectorstore_path))
        logger.info(f"✅ Saved FAISS store to {vectorstore_path} with {len(all_docs)} documents")
        return db
    else:
        raise ValueError("No embeddings were successfully created")

def create_faiss_store_metadata_only(docs: List[Document], topic_id: str):
    """Create a metadata-only store for fast retrieval without embeddings"""
    metadata_path = Path("vectorstores") / str(topic_id) / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    metadata_list = []
    
    for doc in docs:
        metadata_list.append({
            'content': doc.page_content,
            'metadata': doc.metadata
        })
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Saved metadata for {len(docs)} documents to {metadata_path}")
    return len(docs)

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