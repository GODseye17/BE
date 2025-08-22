#!/usr/bin/env python3
"""
GraphRAG Pipeline Test Script
Demonstrates the complete pipeline with timing for each step
"""

import asyncio
import time
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import uuid

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from pubmed.fetcher import fetch_pubmed_data
    from pubmed.filters import PubMedFilters
    from pubmed.query_preprocessor import QueryPreprocessor
    from pipeline.async_pipeline import process_query_async, get_async_pipeline
    from core.globals import get_globals, set_globals
    from vectorstore.manager import get_vectorstore_retriever
    from utils.chains import get_or_create_chain
    from models.requests import TopicRequest, QueryRequest
    from models.responses import TopicResponse, ChatResponse
    from config.settings import (
        SUPABASE_URL, SUPABASE_KEY, TOGETHER_API_KEY, OPENAI_API_KEY,
        LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, EMBEDDING_MODEL
    )
    from llm import TogetherChatModel
    from langchain_huggingface import HuggingFaceEmbeddings
    from supabase import create_client
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class PipelineTimer:
    """Timer class for tracking pipeline step durations"""
    
    def __init__(self):
        self.steps = {}
        self.current_step = None
        self.start_time = None
    
    def start_step(self, step_name: str):
        """Start timing a pipeline step"""
        if self.current_step:
            self.end_step()
        
        self.current_step = step_name
        self.start_time = time.time()
        logger.info(f"🔄 Starting: {step_name}")
    
    def end_step(self):
        """End timing the current step"""
        if self.current_step and self.start_time:
            duration = time.time() - self.start_time
            self.steps[self.current_step] = duration
            logger.info(f"✅ Completed: {self.current_step} ({duration:.2f}s)")
            self.current_step = None
            self.start_time = None
    
    def get_summary(self) -> Dict[str, float]:
        """Get timing summary"""
        self.end_step()  # End any current step
        total_time = sum(self.steps.values())
        return {
            "steps": self.steps,
            "total_time": total_time,
            "step_count": len(self.steps)
        }


class GraphRAGPipelineTester:
    """Main test class for GraphRAG pipeline"""
    
    def __init__(self):
        self.timer = PipelineTimer()
        self.topic_id = None
        self.conversation_id = None
        self.globals_dict = None
        
    async def initialize_system(self):
        """Initialize the system components"""
        self.timer.start_step("System Initialization")
        
        try:
            # Initialize Supabase
            logger.info("🔗 Connecting to Supabase...")
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            
            # Initialize embedding model
            logger.info("🧠 Loading embedding model...")
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            
            # Initialize LLM (if API key available)
            llm = None
            if TOGETHER_API_KEY:
                logger.info("🤖 Loading LLM...")
                try:
                    llm = TogetherChatModel(
                        api_key=TOGETHER_API_KEY,
                        model=LLM_MODEL,
                        temperature=LLM_TEMPERATURE,
                        max_tokens=LLM_MAX_TOKENS,
                        streaming=True
                    )
                    logger.info("✅ LLM loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load LLM: {e}")
            else:
                logger.warning("⚠️ TOGETHER_API_KEY not set - LLM features will be limited")
            
            # Set global instances
            set_globals(
                supabase=supabase,
                llm=llm,
                embeddings=embeddings,
                topic_vectorstores={},
                conversation_chains={},
                background_tasks_status={}
            )
            
            self.globals_dict = get_globals()
            logger.info("✅ System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
    
    async def fetch_articles(self, topic: str, max_results: int = 50) -> str:
        """Fetch articles for the given topic"""
        self.timer.start_step("Article Retrieval")
        
        try:
            logger.info(f"📚 Fetching articles for topic: '{topic}'")
            logger.info(f"📊 Max results: {max_results}")
            
            # Create topic request
            topic_request = TopicRequest(
                topics=[topic],
                operator="AND",
                max_results=max_results,
                filters={
                    "date_from": "2020-01-01",
                    "date_to": datetime.now().strftime("%Y-%m-%d"),
                    "article_types": ["research-article", "review"],
                    "languages": ["eng"]
                }
            )
            
            # Fetch articles
            result = await fetch_pubmed_data(
                topics=[topic],
                operator="AND",
                max_results=max_results,
                filters=topic_request.filters,
                create_embeddings=True
            )
            
            if isinstance(result, dict) and 'topic_id' in result:
                self.topic_id = result['topic_id']
                logger.info(f"📋 Topic ID: {self.topic_id}")
                
                # Wait for processing to complete
                await self.wait_for_topic_processing()
                return self.topic_id
            else:
                raise Exception("Failed to get topic ID from fetch result")
                
        except Exception as e:
            logger.error(f"❌ Article retrieval failed: {e}")
            raise
    
    async def wait_for_topic_processing(self):
        """Wait for topic processing to complete"""
        self.timer.start_step("Topic Processing")
        
        try:
            max_wait_time = 300  # 5 minutes
            wait_interval = 5    # Check every 5 seconds
            elapsed = 0
            
            while elapsed < max_wait_time:
                # Check topic status
                status = await self.check_topic_status()
                
                if status == "completed":
                    logger.info("✅ Topic processing completed")
                    return
                elif status == "failed":
                    raise Exception("Topic processing failed")
                elif status == "processing":
                    logger.info(f"⏳ Processing... ({elapsed}s elapsed)")
                    await asyncio.sleep(wait_interval)
                    elapsed += wait_interval
                else:
                    logger.warning(f"⚠️ Unknown status: {status}")
                    await asyncio.sleep(wait_interval)
                    elapsed += wait_interval
            
            raise Exception(f"Topic processing timeout after {max_wait_time}s")
            
        except Exception as e:
            logger.error(f"❌ Topic processing failed: {e}")
            raise
    
    async def check_topic_status(self) -> str:
        """Check the status of topic processing"""
        try:
            from utils import check_topic_fetch_status
            return check_topic_fetch_status(self.topic_id)
        except Exception as e:
            logger.error(f"Error checking topic status: {e}")
            return "unknown"
    
    async def process_query(self, question: str) -> Dict[str, Any]:
        """Process a query using the GraphRAG pipeline"""
        self.timer.start_step("Query Processing")
        
        try:
            if not self.topic_id:
                raise Exception("No topic ID available. Please fetch articles first.")
            
            logger.info(f"🤔 Processing question: '{question}'")
            
            # Create query request
            query_request = QueryRequest(
                query=question,
                topic_id=self.topic_id,
                conversation_id=self.conversation_id or str(uuid.uuid4())
            )
            
            # Process query using async pipeline
            result = await process_query_async(query_request)
            
            if isinstance(result, dict):
                self.conversation_id = result.get('conversation_id', self.conversation_id)
                return result
            else:
                raise Exception("Invalid result from query processing")
                
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            raise
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        self.timer.start_step("Metrics Collection")
        
        try:
            metrics = {
                "system_info": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "timestamp": datetime.now().isoformat()
                },
                "api_keys": {
                    "together_api": "✅ Set" if TOGETHER_API_KEY else "❌ Missing",
                    "openai_api": "✅ Set" if OPENAI_API_KEY else "❌ Missing",
                    "supabase": "✅ Set" if SUPABASE_URL and SUPABASE_KEY else "❌ Missing"
                },
                "models": {
                    "embedding_model": EMBEDDING_MODEL,
                    "llm_model": LLM_MODEL if TOGETHER_API_KEY else "Not available"
                }
            }
            
            # Add timing summary
            timing_summary = self.timer.get_summary()
            metrics["timing"] = timing_summary
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Metrics collection failed: {e}")
            return {"error": str(e)}
    
    def print_timing_summary(self):
        """Print a formatted timing summary"""
        summary = self.timer.get_summary()
        
        print("\n" + "="*60)
        print("📊 PIPELINE TIMING SUMMARY")
        print("="*60)
        
        for step, duration in summary["steps"].items():
            print(f"⏱️  {step:<25} : {duration:>8.2f}s")
        
        print("-"*60)
        print(f"⏱️  {'TOTAL TIME':<25} : {summary['total_time']:>8.2f}s")
        print(f"📈 {'STEPS COMPLETED':<25} : {summary['step_count']:>8}")
        print("="*60)
    
    async def run_interactive_test(self):
        """Run the interactive test session"""
        print("\n" + "🚀 GRAPHRAG PIPELINE TEST SCRIPT")
        print("="*60)
        print("This script will demonstrate the complete GraphRAG pipeline")
        print("with timing for each step.")
        print("="*60)
        
        try:
            # Step 1: Initialize system
            await self.initialize_system()
            
            # Step 2: Get topic from user
            print("\n📝 STEP 1: Enter a research topic")
            print("-" * 40)
            topic = input("Enter a research topic (e.g., 'cancer immunotherapy'): ").strip()
            
            if not topic:
                print("❌ No topic provided. Exiting.")
                return
            
            # Step 3: Fetch articles
            print(f"\n📚 STEP 2: Fetching articles for '{topic}'")
            print("-" * 40)
            max_results = input("Enter max number of articles (default 50): ").strip()
            max_results = int(max_results) if max_results.isdigit() else 50
            
            await self.fetch_articles(topic, max_results)
            
            # Step 4: Ask questions
            print(f"\n🤔 STEP 3: Ask questions about the articles")
            print("-" * 40)
            print("You can ask multiple questions. Type 'quit' to exit.")
            
            while True:
                question = input("\nEnter your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question:
                    continue
                
                # Process the question
                result = await self.process_query(question)
                
                # Display the answer
                print(f"\n💡 Answer:")
                print("-" * 40)
                if isinstance(result, dict):
                    answer = result.get('answer', 'No answer available')
                    print(answer)
                    
                    # Show additional info if available
                    if 'documents' in result:
                        print(f"\n📄 Sources: {len(result['documents'])} documents referenced")
                    if 'entities' in result:
                        print(f"🏷️  Entities: {len(result['entities'])} entities extracted")
                else:
                    print("❌ Failed to get answer")
            
            # Step 5: Show final metrics
            print(f"\n📊 STEP 4: Final metrics")
            print("-" * 40)
            metrics = await self.get_system_metrics()
            
            # Print timing summary
            self.print_timing_summary()
            
            # Print system metrics
            print(f"\n🔧 System Information:")
            print(f"   Python: {metrics['system_info']['python_version']}")
            print(f"   Platform: {metrics['system_info']['platform']}")
            print(f"   Timestamp: {metrics['system_info']['timestamp']}")
            
            print(f"\n🔑 API Keys Status:")
            for key, status in metrics['api_keys'].items():
                print(f"   {key}: {status}")
            
            print(f"\n🤖 Models:")
            for model, value in metrics['models'].items():
                print(f"   {model}: {value}")
            
            print("\n✅ Test completed successfully!")
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Test interrupted by user")
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            print(f"\n❌ Test failed: {e}")
        finally:
            # Always show timing summary
            self.print_timing_summary()


async def main():
    """Main entry point"""
    tester = GraphRAGPipelineTester()
    await tester.run_interactive_test()


if __name__ == "__main__":
    # Check for required environment variables
    missing_vars = []
    
    if not TOGETHER_API_KEY:
        missing_vars.append("TOGETHER_API_KEY")
    if not SUPABASE_URL or not SUPABASE_KEY:
        missing_vars.append("SUPABASE_URL/SUPABASE_KEY")
    
    if missing_vars:
        print("⚠️  WARNING: Some environment variables are missing:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nThe script will run with limited functionality.")
        print("For full functionality, please set the missing variables.")
        print("\nPress Enter to continue anyway, or Ctrl+C to exit...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
    
    # Run the test
    asyncio.run(main())
