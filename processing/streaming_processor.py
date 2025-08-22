"""
Streaming Document Processor with Memory-Efficient Generators
"""
import gc
import logging
from typing import List, Dict, Any, Generator, Iterator
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

@dataclass
class ProcessingChunk:
    """Container for a chunk of documents being processed"""
    chunk_id: int
    documents: List[Dict[str, Any]]
    processed_count: int
    success_count: int
    error_count: int
    processing_time: float

class StreamingDocumentProcessor:
    """Memory-efficient streaming processor for large document sets"""
    
    def __init__(self, chunk_size: int = 100, max_workers: int = 4):
        """
        Initialize streaming processor
        
        Args:
            chunk_size: Number of documents to process per chunk
            max_workers: Maximum number of worker threads
        """
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.total_processed = 0
        self.total_errors = 0
        
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Split documents into memory-efficient chunks
        
        Args:
            documents: List of documents to chunk
            
        Yields:
            Chunks of documents
        """
        for i in range(0, len(documents), self.chunk_size):
            chunk = documents[i:i + self.chunk_size]
            yield chunk
            
    def process_document_chunk(self, chunk: List[Dict[str, Any]], chunk_id: int) -> ProcessingChunk:
        """
        Process a single chunk of documents
        
        Args:
            chunk: List of documents to process
            chunk_id: Unique identifier for this chunk
            
        Returns:
            ProcessingChunk with results
        """
        start_time = time.time()
        processed_docs = []
        success_count = 0
        error_count = 0
        
        for doc in chunk:
            try:
                # Process document (extract entities, relationships, etc.)
                processed_doc = self._process_single_document(doc)
                processed_docs.append(processed_doc)
                success_count += 1
            except Exception as e:
                logger.error(f"Error processing document {doc.get('pmid', 'unknown')}: {e}")
                error_count += 1
                # Include error document for tracking
                processed_docs.append({
                    **doc,
                    'processing_error': str(e),
                    'processing_status': 'failed'
                })
        
        processing_time = time.time() - start_time
        
        # Force garbage collection after each chunk
        gc.collect()
        
        return ProcessingChunk(
            chunk_id=chunk_id,
            documents=processed_docs,
            processed_count=len(chunk),
            success_count=success_count,
            error_count=error_count,
            processing_time=processing_time
        )
    
    def _process_single_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single document with entity extraction and enhancement
        
        Args:
            doc: Document to process
            
        Returns:
            Enhanced document with extracted entities and relationships
        """
        try:
            # Import processing components
            from knowledge_graph.entity_extractor import MedicalEntityExtractor
            from knowledge_graph.relationship_extractor import RelationshipExtractor
            from pubmed.relevance_scorer import RelevanceScorer
            
            # Initialize extractors (use singleton pattern for efficiency)
            if not hasattr(self, '_entity_extractor'):
                self._entity_extractor = MedicalEntityExtractor()
            if not hasattr(self, '_relationship_extractor'):
                self._relationship_extractor = RelationshipExtractor()
            if not hasattr(self, '_relevance_scorer'):
                self._relevance_scorer = RelevanceScorer()
            
            # Extract text content
            text_content = doc.get('abstract', '') or doc.get('title', '')
            
            # Extract entities
            entities = self._entity_extractor.extract_entities(text_content)
            
            # Extract relationships
            relationships = self._relationship_extractor.extract_relationships(text_content, entities)
            
            # Calculate relevance score
            relevance_score = self._relevance_scorer.score_article(doc)
            
            # Enhanced document
            enhanced_doc = {
                **doc,
                'extracted_entities': entities,
                'extracted_relationships': relationships,
                'relevance_score': relevance_score,
                'processing_status': 'success',
                'processing_timestamp': time.time()
            }
            
            return enhanced_doc
            
        except Exception as e:
            logger.error(f"Error in single document processing: {e}")
            return {
                **doc,
                'processing_error': str(e),
                'processing_status': 'failed',
                'processing_timestamp': time.time()
            }
    
    async def process_documents_async(
        self, 
        documents: List[Dict[str, Any]], 
        callback=None
    ) -> Iterator[ProcessingChunk]:
        """
        Asynchronously process documents in chunks
        
        Args:
            documents: List of documents to process
            callback: Optional callback function for progress updates
            
        Yields:
            ProcessingChunk results as they complete
        """
        logger.info(f"🔄 Starting streaming processing of {len(documents)} documents in chunks of {self.chunk_size}")
        
        # Create chunks
        chunks = list(self.chunk_documents(documents))
        total_chunks = len(chunks)
        
        logger.info(f"📊 Created {total_chunks} chunks for processing")
        
        # Process chunks with concurrency control
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_chunk_async(chunk_data, chunk_id):
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor,
                    self.process_document_chunk,
                    chunk_data,
                    chunk_id
                )
        
        # Submit all chunks for processing
        tasks = [
            process_chunk_async(chunk, i)
            for i, chunk in enumerate(chunks)
        ]
        
        # Process results as they complete
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                chunk_result = await task
                
                # Update counters
                self.total_processed += chunk_result.processed_count
                self.total_errors += chunk_result.error_count
                
                # Call progress callback if provided
                if callback:
                    progress = (i + 1) / total_chunks
                    callback(progress, chunk_result)
                
                logger.info(
                    f"✅ Chunk {chunk_result.chunk_id + 1}/{total_chunks} processed: "
                    f"{chunk_result.success_count} success, {chunk_result.error_count} errors "
                    f"({chunk_result.processing_time:.2f}s)"
                )
                
                yield chunk_result
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                # Yield error chunk for tracking
                yield ProcessingChunk(
                    chunk_id=i,
                    documents=[],
                    processed_count=0,
                    success_count=0,
                    error_count=len(chunks[i]) if i < len(chunks) else 0,
                    processing_time=0.0
                )
        
        # Final cleanup
        gc.collect()
        
        logger.info(
            f"🎯 Streaming processing complete: "
            f"{self.total_processed} documents processed, "
            f"{self.total_errors} errors"
        )
    
    def process_documents_sync(self, documents: List[Dict[str, Any]]) -> List[ProcessingChunk]:
        """
        Synchronously process documents in chunks (fallback method)
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of ProcessingChunk results
        """
        logger.info(f"🔄 Starting synchronous processing of {len(documents)} documents")
        
        results = []
        chunks = list(self.chunk_documents(documents))
        
        for i, chunk in enumerate(chunks):
            try:
                chunk_result = self.process_document_chunk(chunk, i)
                results.append(chunk_result)
                
                self.total_processed += chunk_result.processed_count
                self.total_errors += chunk_result.error_count
                
                logger.info(
                    f"✅ Chunk {i + 1}/{len(chunks)} processed: "
                    f"{chunk_result.success_count} success, {chunk_result.error_count} errors"
                )
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                results.append(ProcessingChunk(
                    chunk_id=i,
                    documents=[],
                    processed_count=0,
                    success_count=0,
                    error_count=len(chunk),
                    processing_time=0.0
                ))
        
        logger.info(f"🎯 Synchronous processing complete: {self.total_processed} documents processed")
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'total_processed': self.total_processed,
            'total_errors': self.total_errors,
            'success_rate': (self.total_processed - self.total_errors) / max(self.total_processed, 1),
            'chunk_size': self.chunk_size,
            'max_workers': self.max_workers
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.total_processed = 0
        self.total_errors = 0
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

class StreamingKnowledgeGraphBuilder:
    """Build knowledge graph from streaming document chunks"""
    
    def __init__(self):
        self.graph_nodes = {}
        self.graph_edges = []
        
    def update_graph_from_chunk(self, chunk: ProcessingChunk):
        """
        Update knowledge graph with entities and relationships from a chunk
        
        Args:
            chunk: ProcessingChunk containing processed documents
        """
        for doc in chunk.documents:
            if doc.get('processing_status') == 'success':
                # Add entities to graph
                entities = doc.get('extracted_entities', [])
                for entity in entities:
                    entity_id = entity.get('id') or entity.get('text', '').lower()
                    if entity_id not in self.graph_nodes:
                        self.graph_nodes[entity_id] = {
                            'type': entity.get('type', 'unknown'),
                            'text': entity.get('text', ''),
                            'frequency': 0,
                            'documents': []
                        }
                    
                    self.graph_nodes[entity_id]['frequency'] += 1
                    self.graph_nodes[entity_id]['documents'].append(doc.get('pmid', ''))
                
                # Add relationships to graph
                relationships = doc.get('extracted_relationships', [])
                for rel in relationships:
                    edge = {
                        'source': rel.get('source', ''),
                        'target': rel.get('target', ''),
                        'type': rel.get('type', ''),
                        'confidence': rel.get('confidence', 0.0),
                        'document': doc.get('pmid', '')
                    }
                    self.graph_edges.append(edge)
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            'total_nodes': len(self.graph_nodes),
            'total_edges': len(self.graph_edges),
            'node_types': list(set(node['type'] for node in self.graph_nodes.values())),
            'edge_types': list(set(edge['type'] for edge in self.graph_edges))
        }

# Global processor instance
_streaming_processor = None

def get_streaming_processor() -> StreamingDocumentProcessor:
    """Get global streaming processor instance"""
    global _streaming_processor
    if _streaming_processor is None:
        _streaming_processor = StreamingDocumentProcessor()
    return _streaming_processor

async def process_articles_streaming(
    articles: List[Dict[str, Any]], 
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Main function to process articles using streaming processor
    
    Args:
        articles: List of articles to process
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of processed articles
    """
    processor = get_streaming_processor()
    
    all_processed_docs = []
    
    async for chunk in processor.process_documents_async(articles, progress_callback):
        all_processed_docs.extend(chunk.documents)
    
    return all_processed_docs