"""
Multi-Agent Coordinator for Orchestrating Specialized Agents
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from .research_agent import ResearchAgent
from .clinical_agent import ClinicalAgent
from .statistical_agent import StatisticalAgent
from .critic_agent import CriticAgent

logger = logging.getLogger(__name__)

class MultiAgentCoordinator:
    """Coordinate multiple specialized agents for comprehensive analysis"""
    
    def __init__(self, together_api_key: str, openai_api_key: str = None):
        """
        Initialize the multi-agent coordinator
        
        Args:
            together_api_key: API key for Together AI (used by 3 agents)
            openai_api_key: API key for OpenAI (used by Critic Agent, optional)
        """
        self.together_api_key = together_api_key
        self.openai_api_key = openai_api_key
        
        # Initialize agents
        self.research_agent = ResearchAgent(together_api_key)
        self.clinical_agent = ClinicalAgent(together_api_key)
        self.statistical_agent = StatisticalAgent(together_api_key)
        self.critic_agent = CriticAgent(openai_api_key)
        
        logger.info("✅ Multi-Agent Coordinator initialized")
        logger.info(f"🔬 Research Agent: Active")
        logger.info(f"👨‍⚕️ Clinical Agent: Active")
        logger.info(f"📊 Statistical Agent: Active")
        logger.info(f"🎯 Critic Agent: {'Active' if openai_api_key else 'Disabled (OpenAI API key required)'}")
    
    async def process_query(self, query: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a query using all available agents
        
        Args:
            query: User's research question
            articles: List of articles to analyze
            
        Returns:
            Comprehensive analysis from all agents
        """
        start_time = datetime.now()
        logger.info(f"🚀 Starting multi-agent analysis for query: {query[:100]}...")
        
        try:
            # Step 1: Parallel processing by specialized agents
            research_task = asyncio.create_task(
                self.research_agent.analyze_literature(articles, query)
            )
            clinical_task = asyncio.create_task(
                self.clinical_agent.assess_clinical_implications({}, articles)  # Empty research findings initially
            )
            statistical_task = asyncio.create_task(
                self.statistical_agent.evaluate_evidence(articles, {})  # Empty research findings initially
            )
            
            # Wait for all agents to complete
            research_analysis, clinical_insights, statistical_evaluation = await asyncio.gather(
                research_task, clinical_task, statistical_task
            )
            
            logger.info("✅ All primary agents completed analysis")
            
            # Step 2: Update clinical and statistical agents with research findings
            if research_analysis and research_analysis.get('analysis'):
                # Re-run clinical and statistical analysis with research context
                clinical_task_v2 = asyncio.create_task(
                    self.clinical_agent.assess_clinical_implications(research_analysis, articles)
                )
                statistical_task_v2 = asyncio.create_task(
                    self.statistical_agent.evaluate_evidence(articles, research_analysis)
                )
                
                clinical_insights, statistical_evaluation = await asyncio.gather(
                    clinical_task_v2, statistical_task_v2
                )
                
                logger.info("✅ Updated analysis with research context completed")
            
            # Step 3: Final validation by critic agent
            validation_result = await self.critic_agent.validate_response(
                research_analysis, clinical_insights, statistical_evaluation
            )
            
            # Step 4: Synthesize final response
            final_response = self._synthesize_response(
                query, articles, research_analysis, clinical_insights, 
                statistical_evaluation, validation_result
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            final_response['processing_time'] = processing_time
            
            logger.info(f"✅ Multi-agent analysis completed in {processing_time:.2f} seconds")
            
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Error in multi-agent processing: {e}")
            return {
                'error': f"Multi-agent processing failed: {str(e)}",
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'query': query,
                'articles_analyzed': len(articles)
            }
    
    def _synthesize_response(self, query: str, articles: List[Dict[str, Any]], 
                           research_analysis: Dict[str, Any], clinical_insights: Dict[str, Any],
                           statistical_evaluation: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final response from all agents"""
        
        # Calculate overall confidence
        research_confidence = research_analysis.get('confidence', 0.0)
        clinical_confidence = clinical_insights.get('clinical_confidence', 0.0)
        statistical_confidence = statistical_evaluation.get('statistical_confidence', 0.0)
        validation_score = validation_result.get('validation_score', 0.0) / 100.0  # Normalize to 0-1
        
        overall_confidence = (research_confidence * 0.3 + 
                            clinical_confidence * 0.25 + 
                            statistical_confidence * 0.25 + 
                            validation_score * 0.2)
        
        # Create comprehensive response
        response = {
            'query': query,
            'articles_analyzed': len(articles),
            'overall_confidence': overall_confidence,
            'analysis_summary': {
                'research_confidence': research_confidence,
                'clinical_confidence': clinical_confidence,
                'statistical_confidence': statistical_confidence,
                'validation_score': validation_score
            },
            'research_analysis': {
                'analysis': research_analysis.get('analysis', ''),
                'insights': research_analysis.get('insights', []),
                'evidence_strength': research_analysis.get('evidence_strength', 'unknown'),
                'research_gaps': research_analysis.get('research_gaps', [])
            },
            'clinical_assessment': {
                'assessment': clinical_insights.get('clinical_assessment', ''),
                'insights': clinical_insights.get('clinical_insights', []),
                'treatment_recommendations': clinical_insights.get('treatment_recommendations', []),
                'safety_considerations': clinical_insights.get('safety_considerations', []),
                'implementation_guidance': clinical_insights.get('implementation_guidance', [])
            },
            'statistical_evaluation': {
                'analysis': statistical_evaluation.get('statistical_analysis', ''),
                'insights': statistical_evaluation.get('statistical_insights', []),
                'effect_sizes': statistical_evaluation.get('effect_sizes', []),
                'evidence_quality': statistical_evaluation.get('evidence_quality', 'unknown'),
                'bias_assessment': statistical_evaluation.get('bias_assessment', [])
            },
            'validation': {
                'report': validation_result.get('validation_report', ''),
                'score': validation_result.get('validation_score', 0.0),
                'issues_found': validation_result.get('issues_found', []),
                'suggestions': validation_result.get('suggestions', []),
                'overall_quality': validation_result.get('overall_quality', 'unknown')
            },
            'key_findings': self._extract_key_findings(
                research_analysis, clinical_insights, statistical_evaluation
            ),
            'recommendations': self._extract_recommendations(
                research_analysis, clinical_insights, statistical_evaluation, validation_result
            ),
            'agent_status': {
                'research_agent': 'active',
                'clinical_agent': 'active',
                'statistical_agent': 'active',
                'critic_agent': 'active' if self.openai_api_key else 'disabled'
            }
        }
        
        return response
    
    def _extract_key_findings(self, research_analysis: Dict[str, Any], 
                            clinical_insights: Dict[str, Any], 
                            statistical_evaluation: Dict[str, Any]) -> List[str]:
        """Extract key findings from all agents"""
        findings = []
        
        # Research findings
        research_insights = research_analysis.get('insights', [])
        findings.extend(research_insights[:3])  # Top 3 research insights
        
        # Clinical findings
        clinical_insights_list = clinical_insights.get('clinical_insights', [])
        findings.extend(clinical_insights_list[:3])  # Top 3 clinical insights
        
        # Statistical findings
        statistical_insights = statistical_evaluation.get('statistical_insights', [])
        findings.extend(statistical_insights[:3])  # Top 3 statistical insights
        
        return findings[:10]  # Limit to top 10 findings
    
    def _extract_recommendations(self, research_analysis: Dict[str, Any], 
                               clinical_insights: Dict[str, Any], 
                               statistical_evaluation: Dict[str, Any],
                               validation_result: Dict[str, Any]) -> List[str]:
        """Extract recommendations from all agents"""
        recommendations = []
        
        # Clinical recommendations
        treatment_recs = clinical_insights.get('treatment_recommendations', [])
        recommendations.extend(treatment_recs)
        
        # Implementation guidance
        implementation_guidance = clinical_insights.get('implementation_guidance', [])
        recommendations.extend(implementation_guidance)
        
        # Validation suggestions
        validation_suggestions = validation_result.get('suggestions', [])
        recommendations.extend(validation_suggestions)
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            'research_agent': {
                'status': 'active',
                'model': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
                'provider': 'Together AI'
            },
            'clinical_agent': {
                'status': 'active',
                'model': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
                'provider': 'Together AI'
            },
            'statistical_agent': {
                'status': 'active',
                'model': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
                'provider': 'Together AI'
            },
            'critic_agent': {
                'status': 'active' if self.openai_api_key else 'disabled',
                'model': 'gpt-4-turbo' if self.openai_api_key else 'none',
                'provider': 'OpenAI' if self.openai_api_key else 'none',
                'note': 'Requires OpenAI API key' if not self.openai_api_key else 'Active'
            }
        }
    
    def enable_critic_agent(self, openai_api_key: str):
        """Enable the critic agent with OpenAI API key"""
        self.openai_api_key = openai_api_key
        self.critic_agent = CriticAgent(openai_api_key)
        logger.info("✅ Critic Agent enabled with OpenAI API key")
    
    def disable_critic_agent(self):
        """Disable the critic agent"""
        self.openai_api_key = None
        self.critic_agent = CriticAgent(None)
        logger.info("⚠️ Critic Agent disabled")
