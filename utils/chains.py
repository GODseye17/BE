"""
Conversation chain management utilities with dynamic prompts
"""
import logging
import re
from typing import Optional
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from core.globals import get_globals
from vectorstore.manager import get_vectorstore_retriever
from utils.prompts import get_dynamic_prompt
from config.settings import MAX_CONVERSATIONS

logger = logging.getLogger(__name__)

class StreamingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for streaming responses"""
    def __init__(self):
        self.tokens = []
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

def check_topic_fetch_status(topic_id: str) -> str:
    """Check if data fetching is complete for a topic"""
    globals_dict = get_globals()
    background_tasks_status = globals_dict['background_tasks_status']
    supabase = globals_dict['supabase']
    
    # First check our internal background task status
    if topic_id in background_tasks_status:
        return background_tasks_status[topic_id]
    
    # Then check in Supabase
    if supabase:
        try:
            result = supabase.table("topics").select("status").eq("id", topic_id).execute()
            if result.data and len(result.data) > 0:
                return result.data[0]["status"]
            return "not_found"
        except Exception as e:
            logger.error(f"Error checking topic status: {str(e)}")
            return f"error: {str(e)}"
    else:
        logger.error("Supabase client not initialized")
        return "database_error"

def create_contextual_compression_retriever(base_retriever, llm, query: str):
    """Create a retriever with contextual compression for better relevance"""
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

def get_or_create_chain(topic_id: str, conversation_id: str, query: str):
    """Get or create a conversation chain with dynamic prompt selection"""
    globals_dict = get_globals()
    conversation_chains = globals_dict['conversation_chains']
    llm = globals_dict['llm']
    
    chain_key = f"{topic_id}:{conversation_id}"
   
    # For existing conversations, we need to update the prompt based on new query
    existing_chain = conversation_chains.get(chain_key)
    
    # Create memory (reuse if exists)
    if existing_chain and hasattr(existing_chain, 'memory'):
        memory = existing_chain.memory
    else:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    # Get retriever
    base_retriever = get_vectorstore_retriever(topic_id, query)
    
    # Add contextual compression for better relevance
    retriever = create_contextual_compression_retriever(base_retriever, llm, query)
    
    # Get dynamic prompt based on the query
    dynamic_prompt = get_dynamic_prompt(query)
    
    logger.info(f"Selected prompt type: {detect_query_type(query)} for query: {query[:50]}...")
    
    # Create the chain with dynamic prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,  # Set to False to reduce noise
        output_key="answer",
        combine_docs_chain_kwargs={
            "prompt": dynamic_prompt,
            "document_separator": "\n\n"  # Better document separation
        },
        response_if_no_docs_found="I don't have specific information about that in the current research papers. Could you rephrase your question or ask about something else from the research?"
    )
    
    # Store the updated chain
    conversation_chains[chain_key] = qa_chain
    
    # Clean up old chains if needed
    if len(conversation_chains) > MAX_CONVERSATIONS:
        oldest_keys = list(conversation_chains.keys())[:-MAX_CONVERSATIONS]
        for key in oldest_keys:
            del conversation_chains[key]
        logger.info(f"🗑️ Cleaned up {len(oldest_keys)} old conversation chains")

    return qa_chain

def detect_query_type(query: str) -> str:
    """Detect query type for logging"""
    q_lower = query.lower()
    
    if any(p in q_lower for p in ['what is', 'define', 'who invented']):
        return 'quick_fact'
    elif any(p in q_lower for p in ['how does', 'mechanism', 'pathway']):
        return 'mechanism'
    elif any(p in q_lower for p in ['methodology', 'study design']):
        return 'methodology'
    elif any(p in q_lower for p in ['treatment', 'clinical', 'therapy']):
        return 'clinical'
    elif any(p in q_lower for p in ['evidence', 'studies show', 'research']):
        return 'synthesis'
    return 'general'

def post_process_response(response: str, query: str) -> str:
    """Post-process the response to ensure quality"""
    # Remove any system prompt leakage
    system_terms = [
        "Available research papers:", "User's question:", "CRITICAL INSTRUCTIONS:",
        "Research papers:", "Papers:", "Query:", "Question:",
        "{context}", "{question}"
    ]
    
    for term in system_terms:
        response = response.replace(term, "")
    
    # Clean up excessive whitespace
    response = re.sub(r'\n{3,}', '\n\n', response)
    response = response.strip()
    
    # Ensure response isn't empty
    if not response or len(response) < 10:
        return "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
    
    return response

def validate_comprehensive_response(query: str, answer: str, topic_id: str) -> str:
    """Validate and enhance responses for comprehensive queries"""
    # Post-process the response first
    answer = post_process_response(answer, query)
    
    # For comprehensive queries, add coverage info if needed
    comprehensive_keywords = ["all articles", "every study", "comprehensive", "systematic review"]
    if any(keyword in query.lower() for keyword in comprehensive_keywords):
        globals_dict = get_globals()
        supabase = globals_dict['supabase']
        
        if supabase:
            try:
                articles_result = supabase.table("articles").select("pubmed_id").eq("topic_id", topic_id).execute()
                expected_count = len(articles_result.data)
                
                # Count unique PMIDs mentioned
                pmids_mentioned = set(re.findall(r'PMID:\s*(\d+)', answer))
                coverage = len(pmids_mentioned)
                
                if coverage < expected_count * 0.5 and expected_count > 5:
                    answer += f"\n\n*Note: This analysis covers {coverage} of {expected_count} available studies. For a complete analysis of all studies, consider breaking down your query into specific aspects.*"
                
            except Exception as e:
                logger.error(f"Error validating comprehensive response: {e}")
    
    return answer