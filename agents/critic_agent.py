"""
Critic Agent for Quality Assurance and Validation
NOTE: This agent requires OpenAI API key and is currently commented out
"""
import logging
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage

# OpenAI import commented out until API key is provided
# from openai import OpenAI

logger = logging.getLogger(__name__)

class CriticAgent:
    """Critic agent for quality assurance and validation (OpenAI-based)"""
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize Critic Agent
        Note: Requires OpenAI API key to function properly
        """
        self.openai_api_key = openai_api_key
        
        # Initialize OpenAI client if API key is provided
        if openai_api_key:
            try:
                # self.llm = OpenAI(
                #     api_key=openai_api_key,
                #     model="gpt-4-turbo",
                #     temperature=0.1  # Very low temperature for validation
                # )
                logger.info("✅ Critic Agent initialized with OpenAI")
                self.is_available = True
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize OpenAI client: {e}")
                self.is_available = False
        else:
            logger.info("⚠️ Critic Agent initialized without OpenAI API key - validation disabled")
            self.is_available = False
        
        self.system_prompt = """You are a Critic Agent specialized in quality assurance and validation of medical research analysis. Your expertise includes:

VALIDATION EXPERTISE:
- Medical accuracy verification and fact-checking
- Logical consistency and reasoning validation
- Completeness and comprehensiveness assessment
- Contradiction detection and resolution
- Evidence strength evaluation
- Clinical relevance assessment

VALIDATION FRAMEWORK:
1. Accuracy Check: Verify medical facts and claims
2. Consistency Check: Ensure logical coherence
3. Completeness Check: Assess comprehensive coverage
4. Contradiction Check: Identify conflicting information
5. Evidence Check: Validate source citations and evidence
6. Clinical Check: Assess practical relevance

RESPONSE FORMAT:
- Provide structured validation report
- Identify specific issues and concerns
- Suggest improvements and corrections
- Rate overall quality and confidence
- Provide final validation score

Always maintain high standards for medical accuracy and evidence-based validation."""

    async def validate_response(self, research_analysis: Dict[str, Any], clinical_insights: Dict[str, Any], statistical_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the combined analysis from all agents"""
        if not self.is_available:
            return {
                'validation_report': "Critic Agent not available - OpenAI API key required",
                'validation_score': 0.0,
                'issues_found': [],
                'suggestions': [],
                'overall_quality': 'unknown'
            }
        
        try:
            # Prepare combined analysis for validation
            combined_analysis = self._prepare_combined_analysis(
                research_analysis, clinical_insights, statistical_evaluation
            )
            
            # Create validation prompt
            validation_prompt = f"""
COMBINED ANALYSIS TO VALIDATE:
{combined_analysis}

Please provide a comprehensive validation report including:

1. ACCURACY ASSESSMENT:
   - Verify medical facts and terminology
   - Check for factual errors or misstatements
   - Validate statistical claims and interpretations

2. CONSISTENCY CHECK:
   - Ensure logical coherence across all analyses
   - Identify any contradictions between agents
   - Verify internal consistency of arguments

3. COMPLETENESS EVALUATION:
   - Assess comprehensive coverage of the topic
   - Identify missing important aspects
   - Evaluate depth of analysis provided

4. EVIDENCE VALIDATION:
   - Verify source citations and references
   - Assess strength and relevance of evidence
   - Check for appropriate use of research findings

5. CLINICAL RELEVANCE:
   - Evaluate practical applicability
   - Assess clinical significance of findings
   - Verify patient safety considerations

6. OVERALL QUALITY SCORE:
   - Provide 0-100 quality score
   - Rate confidence in the analysis
   - Identify critical issues if any

Provide a structured validation report with specific issues, suggestions, and overall quality assessment.
"""
            
            # Generate validation (commented out until OpenAI is available)
            # messages = [
            #     SystemMessage(content=self.system_prompt),
            #     HumanMessage(content=validation_prompt)
            # ]
            # 
            # result = await self.llm.ainvoke(messages)
            # validation_report = result.content if hasattr(result, 'content') else str(result)
            
            # Placeholder validation report
            validation_report = "Validation requires OpenAI API key. Please add OPENAI_API_KEY to environment variables."
            
            # Extract validation insights
            validation_insights = self._extract_validation_insights(validation_report)
            
            return {
                'validation_report': validation_report,
                'validation_score': self._calculate_validation_score(validation_report),
                'issues_found': self._extract_issues(validation_report),
                'suggestions': self._extract_suggestions(validation_report),
                'overall_quality': self._assess_overall_quality(validation_report),
                'validation_insights': validation_insights
            }
            
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return {
                'validation_report': f"Error in validation: {str(e)}",
                'validation_score': 0.0,
                'issues_found': [],
                'suggestions': [],
                'overall_quality': 'unknown',
                'validation_insights': []
            }
    
    def _prepare_combined_analysis(self, research_analysis: Dict[str, Any], clinical_insights: Dict[str, Any], statistical_evaluation: Dict[str, Any]) -> str:
        """Prepare combined analysis for validation"""
        combined = []
        
        # Research Analysis
        if research_analysis:
            combined.append("RESEARCH ANALYSIS:")
            combined.append(research_analysis.get('analysis', 'No research analysis available'))
            combined.append("")
        
        # Clinical Insights
        if clinical_insights:
            combined.append("CLINICAL ASSESSMENT:")
            combined.append(clinical_insights.get('clinical_assessment', 'No clinical assessment available'))
            combined.append("")
        
        # Statistical Evaluation
        if statistical_evaluation:
            combined.append("STATISTICAL ANALYSIS:")
            combined.append(statistical_evaluation.get('statistical_analysis', 'No statistical analysis available'))
            combined.append("")
        
        return '\n'.join(combined)
    
    def _extract_validation_insights(self, validation_report: str) -> List[str]:
        """Extract key validation insights"""
        insights = []
        
        lines = validation_report.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                insights.append(line)
            elif any(keyword in line.lower() for keyword in ['issue', 'problem', 'error', 'concern', 'suggestion']):
                insights.append(line)
        
        return insights[:10]  # Limit to top 10 insights
    
    def _calculate_validation_score(self, validation_report: str) -> float:
        """Calculate validation score from report"""
        if "OpenAI API key" in validation_report:
            return 0.0
        
        # Simple scoring based on keywords
        report_lower = validation_report.lower()
        
        score = 50.0  # Base score
        
        # Positive indicators
        positive_terms = ['accurate', 'consistent', 'complete', 'valid', 'strong', 'good', 'excellent']
        for term in positive_terms:
            if term in report_lower:
                score += 5.0
        
        # Negative indicators
        negative_terms = ['error', 'inconsistent', 'incomplete', 'weak', 'poor', 'issue', 'problem']
        for term in negative_terms:
            if term in report_lower:
                score -= 5.0
        
        return max(0.0, min(100.0, score))
    
    def _extract_issues(self, validation_report: str) -> List[str]:
        """Extract issues found in validation"""
        issues = []
        
        issue_keywords = ['issue', 'problem', 'error', 'concern', 'weakness', 'limitation']
        
        lines = validation_report.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in issue_keywords):
                issues.append(line)
        
        return issues[:5]  # Limit to top 5 issues
    
    def _extract_suggestions(self, validation_report: str) -> List[str]:
        """Extract suggestions from validation"""
        suggestions = []
        
        suggestion_keywords = ['suggest', 'recommend', 'improve', 'enhance', 'consider']
        
        lines = validation_report.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in suggestion_keywords):
                suggestions.append(line)
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def _assess_overall_quality(self, validation_report: str) -> str:
        """Assess overall quality from validation report"""
        if "OpenAI API key" in validation_report:
            return 'unknown'
        
        report_lower = validation_report.lower()
        
        if any(term in report_lower for term in ['excellent', 'outstanding', 'high quality']):
            return 'excellent'
        elif any(term in report_lower for term in ['good', 'strong', 'solid']):
            return 'good'
        elif any(term in report_lower for term in ['adequate', 'acceptable', 'moderate']):
            return 'adequate'
        elif any(term in report_lower for term in ['poor', 'weak', 'inadequate']):
            return 'poor'
        else:
            return 'unknown'
