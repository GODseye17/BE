"""
Research Agent for Literature Analysis and Evidence Synthesis
"""
import logging
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage

from llm import TogetherChatModel

logger = logging.getLogger(__name__)

class ResearchAgent:
    """Research agent specialized in literature analysis and evidence synthesis"""
    
    def __init__(self, together_api_key: str):
        self.llm = TogetherChatModel(
            api_key=together_api_key,
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            temperature=0.3,  # Lower temperature for research accuracy
            max_tokens=2048
        )
        
        self.system_prompt = """You are a Research Agent specialized in medical literature analysis and evidence synthesis. Your expertise includes:

RESEARCH ANALYSIS CAPABILITIES:
- Study methodology evaluation and quality assessment
- Evidence synthesis across multiple studies
- Research gap identification and analysis
- Statistical result interpretation
- Systematic review methodology
- Meta-analysis understanding

ANALYSIS FRAMEWORK:
1. Study Design Assessment: Evaluate methodology, sample size, controls
2. Evidence Quality: Assess bias, confounding, statistical power
3. Result Synthesis: Compare findings across studies
4. Gap Analysis: Identify research limitations and future directions
5. Clinical Relevance: Evaluate practical significance

RESPONSE FORMAT:
- Provide structured analysis with clear sections
- Include confidence levels for findings
- Cite specific studies and evidence
- Highlight methodological strengths/weaknesses
- Identify consensus vs. conflicting evidence

Always maintain scientific rigor and objectivity in your analysis."""

    async def analyze_literature(self, articles: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Analyze literature and synthesize evidence"""
        try:
            # Prepare articles for analysis
            formatted_articles = self._format_articles_for_analysis(articles)
            
            # Create analysis prompt
            analysis_prompt = f"""
QUERY: {query}

ARTICLES TO ANALYZE:
{formatted_articles}

Please provide a comprehensive research analysis including:

1. STUDY OVERVIEW:
   - Number and types of studies analyzed
   - Publication date range and sample sizes
   - Study designs and methodologies

2. EVIDENCE SYNTHESIS:
   - Key findings and their consistency
   - Statistical significance and effect sizes
   - Quality of evidence assessment

3. METHODOLOGICAL ASSESSMENT:
   - Strengths and limitations of studies
   - Risk of bias evaluation
   - Confounding factors identified

4. RESEARCH GAPS:
   - Limitations in current evidence
   - Areas needing further research
   - Unanswered questions

5. EVIDENCE STRENGTH:
   - Overall confidence in findings
   - Consensus vs. conflicting evidence
   - Recommendations for future studies

Provide a structured, evidence-based analysis with specific citations to the articles.
"""
            
            # Generate analysis
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=analysis_prompt)
            ]
            
            result = await self.llm.ainvoke(messages)
            analysis = result.content if hasattr(result, 'content') else str(result)
            
            # Extract key insights
            insights = self._extract_key_insights(analysis, articles)
            
            return {
                'analysis': analysis,
                'insights': insights,
                'confidence': self._calculate_confidence(articles),
                'evidence_strength': self._assess_evidence_strength(articles),
                'research_gaps': self._identify_research_gaps(analysis)
            }
            
        except Exception as e:
            logger.error(f"Error in research analysis: {e}")
            return {
                'analysis': f"Error in research analysis: {str(e)}",
                'insights': [],
                'confidence': 0.0,
                'evidence_strength': 'low',
                'research_gaps': []
            }
    
    def _format_articles_for_analysis(self, articles: List[Dict[str, Any]]) -> str:
        """Format articles for analysis"""
        formatted = []
        
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No title')
            abstract = article.get('abstract', 'No abstract')
            authors = article.get('authors', [])
            pmid = article.get('pmid', 'Unknown')
            publication_date = article.get('publication_date', 'Unknown')
            
            formatted.append(f"""
ARTICLE {i}:
Title: {title}
Authors: {', '.join(authors) if authors else 'Unknown'}
PMID: {pmid}
Date: {publication_date}
Abstract: {abstract}
---""")
        
        return '\n'.join(formatted)
    
    def _extract_key_insights(self, analysis: str, articles: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from analysis"""
        insights = []
        
        # Simple extraction based on common patterns
        lines = analysis.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                insights.append(line)
            elif any(keyword in line.lower() for keyword in ['finding', 'result', 'evidence', 'conclusion']):
                insights.append(line)
        
        return insights[:10]  # Limit to top 10 insights
    
    def _calculate_confidence(self, articles: List[Dict[str, Any]]) -> float:
        """Calculate confidence level based on article quality and quantity"""
        if not articles:
            return 0.0
        
        # Factors affecting confidence
        num_articles = len(articles)
        avg_abstract_length = sum(len(article.get('abstract', '')) for article in articles) / num_articles
        
        # Base confidence from number of articles
        confidence = min(0.9, num_articles * 0.1)
        
        # Adjust based on abstract quality
        if avg_abstract_length > 500:
            confidence += 0.1
        elif avg_abstract_length < 200:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _assess_evidence_strength(self, articles: List[Dict[str, Any]]) -> str:
        """Assess overall evidence strength"""
        if not articles:
            return 'no_evidence'
        
        num_articles = len(articles)
        
        if num_articles >= 10:
            return 'strong'
        elif num_articles >= 5:
            return 'moderate'
        elif num_articles >= 2:
            return 'weak'
        else:
            return 'very_weak'
    
    def _identify_research_gaps(self, analysis: str) -> List[str]:
        """Identify research gaps from analysis"""
        gaps = []
        
        # Look for gap-related keywords
        gap_keywords = [
            'further research', 'more studies', 'limited evidence',
            'research gap', 'unanswered', 'future studies',
            'insufficient data', 'needs investigation', 'requires study'
        ]
        
        lines = analysis.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in gap_keywords):
                gaps.append(line)
        
        return gaps[:5]  # Limit to top 5 gaps
