"""
PubMed Data Fetcher with Performance Optimizations
"""
import asyncio
import aiohttp
import logging
import requests
from typing import List, Optional, Dict, Any
from xml.etree import ElementTree as ET
from langchain.docstore.document import Document
import time

from .filters import PubMedFilters
from .query_preprocessor import QueryPreprocessor
from .article_processor import (
    extract_enhanced_article_data, validate_article_data,
    create_content_chunks
)
from vectorstore.manager import create_faiss_store_in_batches, create_faiss_store_metadata_only
from core.globals import get_globals

logger = logging.getLogger(__name__)

async def fetch_article_details_batch(session: aiohttp.ClientSession, article_ids: List[str], 
                                    batch_size: int = 100) -> str:
    """Fetch article details in batches using async"""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(article_ids[:batch_size]),  # Limit batch size
        "retmode": "xml",
    }
    
    try:
        async with session.get(url, params=params) as response:
            return await response.text()
    except Exception as e:
        logger.error(f"Error fetching batch: {e}")
        # Fallback to sync request
        response = requests.get(url, params=params)
        return response.text

async def fetch_article_details_concurrent(article_ids: List[str]) -> str:
    """Fetch article details using concurrent requests for better performance"""
    batch_size = 100  # PubMed recommends max 100 IDs per request
    
    if len(article_ids) <= batch_size:
        # For small requests, use simple sync request
        params = {
            "db": "pubmed",
            "id": ",".join(article_ids),
            "retmode": "xml",
        }
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.text
    
    # For larger requests, use concurrent fetching
    logger.info(f"Fetching {len(article_ids)} articles in concurrent batches of {batch_size}")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(0, len(article_ids), batch_size):
            batch_ids = article_ids[i:i+batch_size]
            task = fetch_article_details_batch(session, batch_ids, batch_size)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine XML results
        combined_xml = '<?xml version="1.0" ?>\n<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2019//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_190101.dtd">\n<PubmedArticleSet>\n'
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch fetch error: {result}")
                continue
            
            # Parse and extract article elements
            try:
                root = ET.fromstring(result)
                for article in root.findall(".//PubmedArticle"):
                    combined_xml += ET.tostring(article, encoding='unicode')
            except Exception as e:
                logger.error(f"Error parsing batch XML: {e}")
        
        combined_xml += '</PubmedArticleSet>'
        return combined_xml

async def fetch_pubmed_data(topics: Optional[List[str]] = None, operator: str = 'AND',
                           topic: Optional[str] = None, topic_id: str = "", max_results: int = 100,
                           filters: Optional[Dict[str, Any]] = None, 
                           advanced_query: Optional[str] = None,
                           auto_transform: bool = True,
                           create_embeddings: bool = True):
    """Enhanced PubMed data fetcher with performance optimizations"""
    try:
        start_time = time.time()
        
        # Get global instances
        globals_dict = get_globals()
        supabase = globals_dict['supabase']
        
        # Initialize filter builder and query preprocessor
        filter_builder = PubMedFilters()
        query_preprocessor = QueryPreprocessor()
        
        # Transform natural language queries if needed
        if auto_transform and topic and not topics and not advanced_query:
            # Single topic - check if it needs transformation
            if query_preprocessor.looks_like_natural_language(topic):
                original_topic = topic
                topic = query_preprocessor.transform_natural_to_pubmed(topic)
                logger.info(f"🔄 Query transformed: '{original_topic}' -> '{topic}'")
                
                # Store transformation info in Supabase if available
                if supabase and topic_id:
                    try:
                        supabase.table("topics").update({
                            "original_query": original_topic,
                            "transformed_query": topic
                        }).eq("id", topic_id).execute()
                    except Exception as e:
                        logger.warning(f"Failed to store query transformation: {e}")
        
        # Build complete search query with multi-topic support
        search_query = filter_builder.build_complete_query(
            topics=topics,
            operator=operator,
            base_query=topic,  # For backward compatibility
            filters=filters,
            advanced_query=advanced_query
        )
        
        # Log search details
        if topics:
            logger.info(f"🔍 Multi-topic search for: {topics}")
            logger.info(f"🔗 Boolean operator: {operator}")
        elif topic:
            logger.info(f"🔍 Single topic search for: '{topic}'")
        elif advanced_query:
            logger.info(f"🔍 Advanced query search")
        
        logger.info(f"🔍 Final search query: {search_query}")
        logger.info(f"📊 Max results: {max_results}")

        # Step 1: Search PubMed for relevant article IDs
        search_params = {
            "db": "pubmed",
            "term": search_query,
            "retmode": "json",
            "retmax": max_results,
            "sort": filters.get('sort_by', 'relevance') if filters else 'relevance',
            "field": filters.get('search_field', 'title/abstract') if filters else 'title/abstract'
        }
        
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_start = time.time()
        response = requests.get(search_url, params=search_params)
        response.raise_for_status()
        search_time = time.time() - search_start
        logger.info(f"⏱️ Search completed in {search_time:.2f}s")

        search_result = response.json().get("esearchresult", {})
        article_ids = search_result.get("idlist", [])
        total_count = search_result.get("count", "0")
        
        if not article_ids:
            search_description = f"topics {topics} with operator '{operator}'" if topics else f"topic '{topic}'"
            logger.warning(f"⚠️ No articles found for {search_description} with applied filters")
            logger.info(f"📊 Total available articles: {total_count}")
            return False

        logger.info(f"✅ Found {len(article_ids)} articles (total available: {total_count})")

        # Step 2: Fetch article details (optimized with concurrent requests)
        fetch_start = time.time()
        if len(article_ids) > 20:
            logger.info(f"⏳ Fetching {len(article_ids)} articles using concurrent requests...")
            details_response_text = await fetch_article_details_concurrent(article_ids)
        else:
            # For small batches, use simple sync request
            details_params = {
                "db": "pubmed",
                "id": ",".join(article_ids),
                "retmode": "xml",
            }
            details_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            details_response = requests.get(details_url, params=details_params)
            details_response.raise_for_status()
            details_response_text = details_response.text
        
        fetch_time = time.time() - fetch_start
        logger.info(f"⏱️ Article fetch completed in {fetch_time:.2f}s")

        # Step 3: Parse XML with enhanced extraction
        parse_start = time.time()
        root = ET.fromstring(details_response_text)

        docs = []
        articles_data = []
        
        for idx, article in enumerate(root.findall(".//PubmedArticle"), start=1):
            try:
                # Add progress logging every 10 articles
                if idx % 10 == 0:
                    logger.info(f"📄 Processing article {idx}/{len(root.findall('.//PubmedArticle'))}...")
                
                # Extract comprehensive article data
                article_data = extract_enhanced_article_data(article, idx)
                
                # Validate article quality
                if not validate_article_data(article_data):
                    logger.warning(f"⚠️ Skipping low-quality article: {article_data.get('pmid', 'unknown')}")
                    continue

                # Create content chunks for embedding
                content_chunks = create_content_chunks(article_data, topic_id)
                
                # Add chunks as documents
                for chunk in content_chunks:
                    docs.append(Document(
                        page_content=chunk['content'],
                        metadata=chunk['metadata']
                    ))

                articles_data.append({
                    "topic_id": topic_id,
                    "pubmed_id": article_data['pmid'],
                    "title": article_data['title'],
                    "abstract": article_data['abstract'],
                    "authors": article_data['authors'],
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{article_data['pmid']}/"
                })

            except Exception as parse_err:
                logger.warning(f"⚠️ Skipping malformed article: {parse_err}")

        parse_time = time.time() - parse_start
        logger.info(f"⏱️ Article parsing completed in {parse_time:.2f}s")
        logger.info(f"📄 Processed {len(docs)} content chunks from {len(articles_data)} articles.")

        if not docs:
            search_description = f"topics {topics}" if topics else f"topic '{topic}'"
            logger.warning(f"⚠️ No valid articles to process for {search_description}")
            return False

        # Step 4: Create vector store (with optimized batching)
        if create_embeddings:
            try:
                embedding_start = time.time()
                # Use optimized batch processing with larger batch size
                db = create_faiss_store_in_batches(docs, topic_id, batch_size=50)
                embedding_time = time.time() - embedding_start
                logger.info(f"⏱️ Embeddings created in {embedding_time:.2f}s")
                logger.info(f"✅ Created topic-specific FAISS store for topic {topic_id} with {len(docs)} chunks")
            except Exception as e:
                logger.error(f"❌ Vector store error: {e}")
                raise  # Re-raise to mark the fetch as failed
        else:
            # Metadata-only mode for faster processing
            metadata_start = time.time()
            create_faiss_store_metadata_only(docs, topic_id)
            metadata_time = time.time() - metadata_start
            logger.info(f"⏱️ Metadata saved in {metadata_time:.2f}s (embeddings skipped)")

        # Step 5: Store metadata to Supabase
        try:
            if supabase:
                supabase.table("topics").update({
                    "status": "completed",
                    "total_articles_found": total_count,
                    "article_count": len(articles_data),
                    "search_topics": topics,
                    "boolean_operator": operator,
                    "filters": filters,
                    "embeddings_created": create_embeddings
                }).eq("id", topic_id).execute()

                supabase.table("articles").insert(articles_data).execute()
                logger.info(f"✅ Stored {len(articles_data)} articles metadata to Supabase.")
        except Exception as e:
            logger.warning(f"⚠️ Supabase error: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"⏱️ Total processing time: {total_time:.2f}s")
        logger.info(f"✅ Processing completed with {len(docs)} chunks for topic_id '{topic_id}'")
        return True

    except Exception as e:
        logger.error(f"❌ Error fetching PubMed data: {e}")
        try:
            if supabase:
                supabase.table("topics").update({
                    "status": f"error: {str(e)}", 
                    "article_count": 0
                }).eq("id", topic_id).execute()
        except:
            pass
        return False