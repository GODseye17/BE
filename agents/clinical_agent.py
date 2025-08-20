"""
Clinical Agent for Clinical Implications and Patient Care Recommendations
"""
import logging
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage

from llm import TogetherChatModel

logger = logging.getLogger(__name__)

class ClinicalAgent:
    """Clinical agent specialized in clinical implications and patient care"""
    
    def __init__(self, together_api_key: str):
        self.llm = TogetherChatModel(
            api_key=together_api_key,
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            temperature=0.4,  # Balanced temperature for clinical reasoning
            max_tokens=2048
        )
        
        self.system_prompt = """You are a Clinical Agent specialized in translating research findings into clinical practice and patient care recommendations. Your expertise includes:

CLINICAL EXPERTISE:
- Evidence-based medicine and clinical guidelines
- Patient care recommendations and treatment protocols
- Risk-benefit analysis for clinical interventions
- Patient safety and contraindication assessment
- Clinical decision-making frameworks
- Healthcare quality and outcomes

CLINICAL ANALYSIS FRAMEWORK:
1. Clinical Relevance: Assess practical significance of findings
2. Patient Impact: Evaluate effects on patient outcomes
3. Treatment Implications: Identify clinical applications
4. Safety Assessment: Consider risks and contraindications
5. Implementation Guidance: Provide practical recommendations

RESPONSE FORMAT:
- Focus on clinical applicability and patient care
- Provide actionable recommendations
- Include safety considerations and contraindications
- Address different patient populations when relevant
- Consider healthcare system implications

Always prioritize patient safety and evidence-based practice in your recommendations."""

    async def assess_clinical_implications(self, research_findings: Dict[str, Any], articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess clinical implications of research findings"""
        try:
            # Extract research analysis
            research_analysis = research_findings.get('analysis', '')
            
            # Prepare articles for clinical assessment
            formatted_articles = self._format_articles_for_clinical_assessment(articles)
            
            # Create clinical assessment prompt
            clinical_prompt = f"""
RESEARCH ANALYSIS:
{research_analysis}

ARTICLES ANALYZED:
{formatted_articles}

Please provide a comprehensive clinical assessment including:

1. CLINICAL RELEVANCE:
   - Practical significance of findings for patient care
   - Applicability to different patient populations
   - Clinical decision-making implications

2. TREATMENT IMPLICATIONS:
   - Potential treatment modifications or new approaches
   - Drug therapy recommendations
   - Procedural or intervention guidance
   - Preventive care implications

3. PATIENT SAFETY CONSIDERATIONS:
   - Potential risks and adverse effects
   - Contraindications and precautions
   - Monitoring requirements
   - Drug interactions or complications

4. CLINICAL GUIDELINES:
   - Alignment with current clinical guidelines
   - Recommendations for guideline updates
   - Quality improvement opportunities

5. IMPLEMENTATION RECOMMENDATIONS:
   - Practical steps for clinical implementation
   - Healthcare system considerations
   - Resource requirements and cost implications
   - Training and education needs

6. PATIENT EDUCATION:
   - Key points for patient communication
   - Shared decision-making considerations
   - Patient engagement strategies

Provide evidence-based clinical recommendations with clear actionability and safety considerations.
"""
            
            # Generate clinical assessment
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=clinical_prompt)
            ]
            
            result = await self.llm.ainvoke(messages)
            clinical_assessment = result.content if hasattr(result, 'content') else str(result)
            
            # Extract clinical insights
            clinical_insights = self._extract_clinical_insights(clinical_assessment)
            
            return {
                'clinical_assessment': clinical_assessment,
                'clinical_insights': clinical_insights,
                'treatment_recommendations': self._extract_treatment_recommendations(clinical_assessment),
                'safety_considerations': self._extract_safety_considerations(clinical_assessment),
                'implementation_guidance': self._extract_implementation_guidance(clinical_assessment),
                'clinical_confidence': self._calculate_clinical_confidence(articles, research_findings)
            }
            
        except Exception as e:
            logger.error(f"Error in clinical assessment: {e}")
            return {
                'clinical_assessment': f"Error in clinical assessment: {str(e)}",
                'clinical_insights': [],
                'treatment_recommendations': [],
                'safety_considerations': [],
                'implementation_guidance': [],
                'clinical_confidence': 0.0
            }
    
    def _format_articles_for_clinical_assessment(self, articles: List[Dict[str, Any]]) -> str:
        """Format articles for clinical assessment"""
        formatted = []
        
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No title')
            abstract = article.get('abstract', 'No abstract')
            publication_date = article.get('publication_date', 'Unknown')
            study_type = self._identify_study_type(abstract)
            
            formatted.append(f"""
STUDY {i}:
Type: {study_type}
Title: {title}
Date: {publication_date}
Abstract: {abstract}
---""")
        
        return '\n'.join(formatted)
    
    def _identify_study_type(self, abstract: str) -> str:
        """Identify study type from abstract"""
        abstract_lower = abstract.lower()
        
        if any(term in abstract_lower for term in ['randomized', 'randomised', 'rct', 'clinical trial']):
            return 'Randomized Controlled Trial'
        elif any(term in abstract_lower for term in ['systematic review', 'meta-analysis']):
            return 'Systematic Review/Meta-analysis'
        elif any(term in abstract_lower for term in ['cohort study', 'prospective']):
            return 'Cohort Study'
        elif any(term in abstract_lower for term in ['case-control', 'retrospective']):
            return 'Case-Control Study'
        elif any(term in abstract_lower for term in ['case report', 'case series']):
            return 'Case Report/Series'
        else:
            return 'Observational Study'
    
    def _extract_clinical_insights(self, clinical_assessment: str) -> List[str]:
        """Extract key clinical insights"""
        insights = []
        
        lines = clinical_assessment.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                insights.append(line)
            elif any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'indicate', 'clinical', 'patient']):
                insights.append(line)
        
        return insights[:10]  # Limit to top 10 insights
    
    def _extract_treatment_recommendations(self, clinical_assessment: str) -> List[str]:
        """Extract treatment recommendations"""
        recommendations = []
        
        treatment_keywords = [
            'treatment', 'therapy', 'medication', 'drug', 'intervention',
            'procedure', 'surgery', 'management', 'approach', 'protocol'
        ]
        
        lines = clinical_assessment.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in treatment_keywords):
                recommendations.append(line)
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _extract_safety_considerations(self, clinical_assessment: str) -> List[str]:
        """Extract safety considerations"""
        safety_points = []
        
        safety_keywords = [
            'safety', 'risk', 'adverse', 'side effect', 'contraindication',
            'precaution', 'monitoring', 'complication', 'toxicity'
        ]
        
        lines = clinical_assessment.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in safety_keywords):
                safety_points.append(line)
        
        return safety_points[:5]  # Limit to top 5 safety points
    
    def _extract_implementation_guidance(self, clinical_assessment: str) -> List[str]:
        """Extract implementation guidance"""
        guidance = []
        
        implementation_keywords = [
            'implement', 'practice', 'guideline', 'protocol', 'standard',
            'workflow', 'process', 'procedure', 'algorithm', 'decision'
        ]
        
        lines = clinical_assessment.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in implementation_keywords):
                guidance.append(line)
        
        return guidance[:5]  # Limit to top 5 guidance points
    
    def _calculate_clinical_confidence(self, articles: List[Dict[str, Any]], research_findings: Dict[str, Any]) -> float:
        """Calculate clinical confidence level"""
        if not articles:
            return 0.0
        
        # Base confidence from research findings
        research_confidence = research_findings.get('confidence', 0.0)
        
        # Adjust based on clinical relevance of studies
        clinical_relevance_score = 0.0
        for article in articles:
            abstract = article.get('abstract', '').lower()
            title = article.get('title', '').lower()
            
            # Clinical relevance indicators
            clinical_terms = [
                'clinical', 'patient', 'treatment', 'therapy', 'outcome',
                'efficacy', 'effectiveness', 'safety', 'trial', 'intervention'
            ]
            
            if any(term in abstract or term in title for term in clinical_terms):
                clinical_relevance_score += 0.1
        
        clinical_relevance_score = min(0.3, clinical_relevance_score)
        
        # Combine research and clinical confidence
        total_confidence = (research_confidence * 0.7) + clinical_relevance_score
        
        return max(0.0, min(1.0, total_confidence))
