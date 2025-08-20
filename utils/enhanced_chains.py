"""
Enhanced Chains with Knowledge Graph and Multi-Agent System Integration
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain

from core import get_globals
from config.settings import TOGETHER_API_KEY, OPENAI_API_KEY
from knowledge_graph import MedicalKnowledgeGraph, GraphRetriever
from agents import MultiAgentCoordinator
from utils.monitoring import PerformanceMonitor

logger = logging.getLogger(__name__)

# Global instances
_knowledge_graphs = {}
_multi_agent_coordinators = {}
performance_monitor = PerformanceMonitor()  # Added performance monitoring

@performance_monitor.track_performance("enhanced_chain_creation")
def get_or_create_enhanced_chain(topic_id: str, conversation_id: str, query: str):
    """Get or create an enhanced conversation chain with knowledge graph and multi-agent system"""
    globals_dict = get_globals()
    llm = globals_dict['llm']
    embeddings = globals_dict['embeddings']
    conversation_chains = globals_dict['conversation_chains']
    
    if not llm or not embeddings:
        logger.error("LLM or embeddings not available")
        return None
    
    # Create unique chain key
    chain_key = f"{topic_id}_{conversation_id}"
    
    # Check if chain already exists
    if chain_key in conversation_chains:
        logger.info(f"📋 Using existing enhanced chain for {chain_key}")
        return conversation_chains[chain_key]
    
    logger.info(f"🔧 Creating new enhanced chain for {chain_key}")
    
    # Initialize knowledge graph if not exists
    if topic_id not in _knowledge_graphs:
        _knowledge_graphs[topic_id] = MedicalKnowledgeGraph()
        logger.info(f"🧠 Initialized knowledge graph for topic {topic_id}")
    
    # Initialize multi-agent coordinator if not exists
    if topic_id not in _multi_agent_coordinators:
        _multi_agent_coordinators[topic_id] = MultiAgentCoordinator(
            together_api_key=TOGETHER_API_KEY,
            openai_api_key=OPENAI_API_KEY
        )
        logger.info(f"🤖 Initialized multi-agent coordinator for topic {topic_id}")
    
    # Get base retriever
    base_retriever = get_vectorstore_retriever(topic_id, query)
    
    # Create graph-enhanced retriever
    graph_retriever = GraphRetriever(_knowledge_graphs[topic_id])
    
    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(base_retriever, graph_retriever)
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Enhanced prompt template
    enhanced_prompt_template = """You are an advanced medical research assistant with access to a comprehensive knowledge graph and multi-agent analysis system. Use the following context to answer the user's question.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

USER QUESTION: {question}

KNOWLEDGE GRAPH INSIGHTS:
- Use entity relationships and medical connections
- Consider clinical implications and evidence quality
- Reference statistical significance and bias assessment

RESPONSE GUIDELINES:
1. Provide evidence-based answers with specific citations
2. Include clinical relevance and practical implications
3. Address statistical confidence and limitations
4. Consider safety considerations and contraindications
5. Suggest areas for further research when appropriate

ANSWER:"""

    # Create question generator chain
    question_generator_prompt = """Given the following conversation history and a new question, rephrase the new question to be a standalone question that captures all relevant context.

Chat History:
{chat_history}

New Question: {question}

Standalone Question:"""

    question_generator = LLMChain(llm=llm, prompt=question_generator_prompt)
    
    # Create the enhanced conversational chain
    enhanced_chain = EnhancedConversationalChain(
        retriever=hybrid_retriever,
        memory=memory,
        question_generator=question_generator,
        llm=llm,
        prompt_template=enhanced_prompt_template,
        multi_agent_coordinator=_multi_agent_coordinators[topic_id],
        knowledge_graph=_knowledge_graphs[topic_id]
    )
    
    # Store the chain
    conversation_chains[chain_key] = enhanced_chain
    
    logger.info(f"✅ Enhanced chain created for {chain_key}")
    return enhanced_chain

class HybridRetriever:
    """Hybrid retriever combining vector search and knowledge graph"""
    
    def __init__(self, vector_retriever, graph_retriever):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
    
    @performance_monitor.track_performance("hybrid_retrieval")
    def get_relevant_documents(self, query: str):
        """Get documents using both vector search and knowledge graph"""
        # Get documents from vector retriever
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        
        # Get graph-enhanced documents
        graph_docs = self.graph_retriever.graph_search(query, self._docs_to_dicts(vector_docs))
        
        # Combine and deduplicate
        combined_docs = self._combine_documents(vector_docs, graph_docs)
        
        return combined_docs
    
    def _docs_to_dicts(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Convert Document objects to dictionaries"""
        return [
            {
                'page_content': doc.page_content,
                'metadata': doc.metadata
            }
            for doc in docs
        ]
    
    def _combine_documents(self, vector_docs: List[Document], graph_docs: List[Dict[str, Any]]) -> List[Document]:
        """Combine and deduplicate documents"""
        # Create a set of document IDs to avoid duplicates
        seen_ids = set()
        combined_docs = []
        
        # Add vector documents
        for doc in vector_docs:
            doc_id = doc.metadata.get('pmid', doc.page_content[:100])
            if doc_id not in seen_ids:
                combined_docs.append(doc)
                seen_ids.add(doc_id)
        
        # Add graph-enhanced documents
        for doc_dict in graph_docs:
            doc_id = doc_dict.get('metadata', {}).get('pmid', doc_dict['page_content'][:100])
            if doc_id not in seen_ids:
                doc = Document(
                    page_content=doc_dict['page_content'],
                    metadata=doc_dict['metadata']
                )
                combined_docs.append(doc)
                seen_ids.add(doc_id)
        
        return combined_docs

class EnhancedConversationalChain:
    """Enhanced conversational chain with multi-agent analysis"""
    
    def __init__(self, retriever, memory, question_generator, llm, prompt_template, 
                 multi_agent_coordinator, knowledge_graph):
        self.retriever = retriever
        self.memory = memory
        self.question_generator = question_generator
        self.llm = llm
        self.prompt_template = prompt_template
        self.multi_agent_coordinator = multi_agent_coordinator
        self.knowledge_graph = knowledge_graph
    
    @performance_monitor.track_performance("enhanced_query_processing")
    async def ainvoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously invoke the enhanced chain"""
        question = inputs.get("question", "")
        
        # Get relevant documents
        docs = self.retriever.get_relevant_documents(question)
        
        # Convert documents to articles format for multi-agent analysis
        articles = self._docs_to_articles(docs)
        
        try:
            # Run multi-agent analysis with timeout
            multi_agent_result = await asyncio.wait_for(
                self.multi_agent_coordinator.process_query(question, articles),
                timeout=30.0  # 30 second timeout
            )
            
            # Create enhanced context with multi-agent insights
            enhanced_context = self._create_enhanced_context(docs, multi_agent_result)
            
            # Generate response using enhanced context
            response = await self._generate_response(question, enhanced_context, multi_agent_result)
            
            return {
                "answer": response,
                "source_documents": docs,
                "multi_agent_analysis": multi_agent_result
            }
        except asyncio.TimeoutError:
            logger.error("Multi-agent processing timed out")
            return {
                "answer": "I apologize, but the analysis is taking longer than expected. Please try a more specific question.",
                "source_documents": [],
                "multi_agent_analysis": {"error": "timeout"}
            }
        except Exception as e:
            logger.error(f"Enhanced chain processing failed: {e}")
            return {
                "answer": "I encountered an error while processing your request. Please try again.",
                "source_documents": [],
                "multi_agent_analysis": {"error": str(e)}
            }
    
    def _docs_to_articles(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Convert Document objects to article format"""
        articles = []
        
        for doc in docs:
            metadata = doc.metadata
            article = {
                'title': metadata.get('title', ''),
                'abstract': doc.page_content,
                'pmid': metadata.get('pmid', ''),
                'authors': metadata.get('authors', []),
                'publication_date': metadata.get('publication_date', ''),
                'relevance_score': metadata.get('relevance_score', 0.0)
            }
            articles.append(article)
        
        return articles
    
    def _create_enhanced_context(self, docs: List[Document], multi_agent_result: Dict[str, Any]) -> str:
        """Create enhanced context with multi-agent insights"""
        # Base context from documents
        base_context = "\n\n".join([doc.page_content for doc in docs])
        
        # Add multi-agent insights
        insights = []
        
        # Research insights
        research_insights = multi_agent_result.get('research_analysis', {}).get('insights', [])
        if research_insights:
            insights.append("RESEARCH INSIGHTS:\n" + "\n".join(research_insights[:3]))
        
        # Clinical insights
        clinical_insights = multi_agent_result.get('clinical_assessment', {}).get('insights', [])
        if clinical_insights:
            insights.append("CLINICAL INSIGHTS:\n" + "\n".join(clinical_insights[:3]))
        
        # Statistical insights
        statistical_insights = multi_agent_result.get('statistical_evaluation', {}).get('insights', [])
        if statistical_insights:
            insights.append("STATISTICAL INSIGHTS:\n" + "\n".join(statistical_insights[:3]))
        
        # Combine context
        enhanced_context = base_context
        if insights:
            enhanced_context += "\n\n" + "\n\n".join(insights)
        
        return enhanced_context
    
    async def _generate_response(self, question: str, context: str, multi_agent_result: Dict[str, Any]) -> str:
        """Generate response using enhanced context and multi-agent analysis"""
        try:
            # Create prompt with enhanced context
            prompt = self.prompt_template.format(
                context=context,
                chat_history="",  # Could be enhanced with memory
                question=question
            )
            
            # Generate response using LLM with timeout
            messages = [{"role": "user", "content": prompt}]
            result = await asyncio.wait_for(
                self.llm.ainvoke(messages),
                timeout=15.0  # 15 second timeout for LLM
            )
            
            response = result.content if hasattr(result, 'content') else str(result)
            
            # Add confidence and quality indicators
            overall_confidence = multi_agent_result.get('overall_confidence', 0.0)
            evidence_quality = multi_agent_result.get('statistical_evaluation', {}).get('evidence_quality', 'unknown')
            
            confidence_note = f"\n\n[Confidence: {overall_confidence:.1%} | Evidence Quality: {evidence_quality.upper()}]"
            response += confidence_note
            
            return response
        except asyncio.TimeoutError:
            return "I apologize, but the response generation is taking longer than expected. Please try a more specific question."
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I encountered an error while generating the response. Please try again."

def get_vectorstore_retriever(topic_id: str, query: str):
    """Get vector store retriever (imported from existing chains)"""
    from utils.chains import get_vectorstore_retriever as get_base_retriever
    return get_base_retriever(topic_id, query)

@performance_monitor.track_performance("knowledge_graph_building")
def build_knowledge_graph_for_topic(topic_id: str, articles: List[Dict[str, Any]]):
    """Build knowledge graph for a specific topic"""
    if topic_id not in _knowledge_graphs:
        _knowledge_graphs[topic_id] = MedicalKnowledgeGraph()
    
    knowledge_graph = _knowledge_graphs[topic_id]
    graph = knowledge_graph.build_from_articles(articles)
    
    logger.info(f"🧠 Knowledge graph built for topic {topic_id}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    return graph

def get_knowledge_graph_statistics(topic_id: str) -> Dict[str, Any]:
    """Get statistics for a topic's knowledge graph"""
    if topic_id not in _knowledge_graphs:
        return {"error": "Knowledge graph not found for topic"}
    
    return _knowledge_graphs[topic_id].get_graph_statistics()

async def get_multi_agent_status(topic_id: str) -> Dict[str, Any]:
    """Get status of multi-agent system for a topic"""
    if topic_id not in _multi_agent_coordinators:
        return {"error": "Multi-agent coordinator not found for topic"}
    
    return await _multi_agent_coordinators[topic_id].get_agent_status()

def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics"""
    return performance_monitor.get_metrics()
