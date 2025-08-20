"""
Statistical Agent for Statistical Analysis and Evidence Grading
"""
import logging
import re
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage

from llm import TogetherChatModel

logger = logging.getLogger(__name__)

class StatisticalAgent:
    """Statistical agent specialized in statistical analysis and evidence grading"""
    
    def __init__(self, together_api_key: str):
        self.llm = TogetherChatModel(
            api_key=together_api_key,
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            temperature=0.2,  # Very low temperature for statistical precision
            max_tokens=2048
        )
        
        self.system_prompt = """You are a Statistical Agent specialized in statistical analysis, evidence grading, and meta-analysis. Your expertise includes:

STATISTICAL EXPERTISE:
- Statistical significance testing and p-value interpretation
- Effect size calculation and interpretation (Cohen's d, odds ratios, risk ratios)
- Confidence intervals and margin of error analysis
- Meta-analysis and systematic review statistics
- GRADE methodology for evidence quality assessment
- Bias assessment and risk of bias evaluation
- Sample size and power analysis
- Statistical heterogeneity and publication bias

EVIDENCE GRADING FRAMEWORK (GRADE):
- Study Design: RCTs (high) → Observational (low) → Case reports (very low)
- Risk of Bias: Low → Moderate → High → Very High
- Inconsistency: No → Serious → Very Serious
- Indirectness: No → Serious → Very Serious
- Imprecision: No → Serious → Very Serious
- Publication Bias: Unlikely → Likely → Very Likely

STATISTICAL ANALYSIS FRAMEWORK:
1. Effect Size Assessment: Calculate and interpret effect sizes
2. Statistical Significance: Evaluate p-values and confidence intervals
3. Clinical Significance: Assess practical importance of findings
4. Heterogeneity Analysis: Evaluate consistency across studies
5. Bias Assessment: Identify and quantify various biases
6. Evidence Quality: Apply GRADE methodology

RESPONSE FORMAT:
- Provide quantitative statistical analysis
- Include effect sizes and confidence intervals
- Apply GRADE methodology for evidence quality
- Identify statistical limitations and biases
- Provide meta-analysis when appropriate

Always maintain statistical rigor and provide precise quantitative assessments."""

    async def evaluate_evidence(self, studies: List[Dict[str, Any]], research_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate statistical evidence and apply GRADE methodology"""
        try:
            # Extract research analysis
            research_analysis = research_findings.get('analysis', '')
            
            # Prepare studies for statistical analysis
            formatted_studies = self._format_studies_for_statistical_analysis(studies)
            
            # Create statistical analysis prompt
            statistical_prompt = f"""
RESEARCH ANALYSIS:
{research_analysis}

STUDIES TO ANALYZE:
{formatted_studies}

Please provide a comprehensive statistical analysis including:

1. EFFECT SIZE ANALYSIS:
   - Calculate and interpret effect sizes (Cohen's d, odds ratios, risk ratios)
   - Assess clinical significance vs. statistical significance
   - Evaluate confidence intervals and precision

2. STATISTICAL SIGNIFICANCE:
   - Interpret p-values and significance levels
   - Assess multiple testing and false discovery rates
   - Evaluate statistical power and sample size adequacy

3. HETEROGENEITY ASSESSMENT:
   - Evaluate consistency across studies (I² statistic)
   - Identify sources of heterogeneity
   - Assess subgroup analysis validity

4. BIAS ASSESSMENT:
   - Publication bias evaluation (funnel plots, Egger's test)
   - Selection bias and reporting bias
   - Performance bias and detection bias
   - Attrition bias and reporting bias

5. GRADE EVIDENCE QUALITY:
   - Study design assessment (RCT vs. observational)
   - Risk of bias evaluation
   - Inconsistency assessment
   - Indirectness evaluation
   - Imprecision assessment
   - Publication bias evaluation
   - Overall evidence quality grade (High/Moderate/Low/Very Low)

6. META-ANALYSIS (if applicable):
   - Pooled effect estimates
   - Forest plot interpretation
   - Sensitivity analysis
   - Subgroup analysis

7. STATISTICAL LIMITATIONS:
   - Identify statistical weaknesses
   - Recommend improvements for future studies
   - Address missing data and handling

Provide quantitative statistical analysis with precise effect sizes, confidence intervals, and GRADE evidence quality assessment.
"""
            
            # Generate statistical analysis
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=statistical_prompt)
            ]
            
            result = await self.llm.ainvoke(messages)
            statistical_analysis = result.content if hasattr(result, 'content') else str(result)
            
            # Extract statistical insights
            statistical_insights = self._extract_statistical_insights(statistical_analysis)
            
            return {
                'statistical_analysis': statistical_analysis,
                'statistical_insights': statistical_insights,
                'effect_sizes': self._extract_effect_sizes(statistical_analysis),
                'evidence_quality': self._extract_evidence_quality(statistical_analysis),
                'bias_assessment': self._extract_bias_assessment(statistical_analysis),
                'statistical_confidence': self._calculate_statistical_confidence(studies, statistical_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            return {
                'statistical_analysis': f"Error in statistical analysis: {str(e)}",
                'statistical_insights': [],
                'effect_sizes': [],
                'evidence_quality': 'very_low',
                'bias_assessment': [],
                'statistical_confidence': 0.0
            }
    
    def _format_studies_for_statistical_analysis(self, studies: List[Dict[str, Any]]) -> str:
        """Format studies for statistical analysis"""
        formatted = []
        
        for i, study in enumerate(studies, 1):
            title = study.get('title', 'No title')
            abstract = study.get('abstract', 'No abstract')
            publication_date = study.get('publication_date', 'Unknown')
            
            # Extract statistical information from abstract
            statistical_info = self._extract_statistical_info(abstract)
            
            formatted.append(f"""
STUDY {i}:
Title: {title}
Date: {publication_date}
Statistical Information: {statistical_info}
Abstract: {abstract}
---""")
        
        return '\n'.join(formatted)
    
    def _extract_statistical_info(self, abstract: str) -> str:
        """Extract statistical information from abstract"""
        statistical_terms = []
        
        # Look for p-values
        p_values = re.findall(r'p\s*[<>=]\s*0\.\d+', abstract, re.IGNORECASE)
        if p_values:
            statistical_terms.extend(p_values)
        
        # Look for confidence intervals
        ci_patterns = re.findall(r'\d+\.?\d*\s*\([^)]*\)', abstract)
        if ci_patterns:
            statistical_terms.extend(ci_patterns[:3])  # Limit to first 3
        
        # Look for effect sizes
        effect_patterns = re.findall(r'(?:odds ratio|risk ratio|hazard ratio|relative risk|effect size|cohen\'s d)\s*[=:]\s*\d+\.?\d*', abstract, re.IGNORECASE)
        if effect_patterns:
            statistical_terms.extend(effect_patterns)
        
        # Look for sample sizes
        sample_patterns = re.findall(r'n\s*=\s*\d+', abstract, re.IGNORECASE)
        if sample_patterns:
            statistical_terms.extend(sample_patterns)
        
        return '; '.join(statistical_terms) if statistical_terms else 'No statistical information found'
    
    def _extract_statistical_insights(self, statistical_analysis: str) -> List[str]:
        """Extract key statistical insights"""
        insights = []
        
        lines = statistical_analysis.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                insights.append(line)
            elif any(keyword in line.lower() for keyword in ['significant', 'effect', 'bias', 'confidence', 'p-value']):
                insights.append(line)
        
        return insights[:10]  # Limit to top 10 insights
    
    def _extract_effect_sizes(self, statistical_analysis: str) -> List[str]:
        """Extract effect size information"""
        effect_sizes = []
        
        effect_keywords = [
            'effect size', 'odds ratio', 'risk ratio', 'hazard ratio',
            'relative risk', 'cohen\'s d', 'correlation', 'regression'
        ]
        
        lines = statistical_analysis.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in effect_keywords):
                effect_sizes.append(line)
        
        return effect_sizes[:5]  # Limit to top 5 effect sizes
    
    def _extract_evidence_quality(self, statistical_analysis: str) -> str:
        """Extract evidence quality grade"""
        analysis_lower = statistical_analysis.lower()
        
        if 'high quality' in analysis_lower or 'grade: high' in analysis_lower:
            return 'high'
        elif 'moderate quality' in analysis_lower or 'grade: moderate' in analysis_lower:
            return 'moderate'
        elif 'low quality' in analysis_lower or 'grade: low' in analysis_lower:
            return 'low'
        elif 'very low quality' in analysis_lower or 'grade: very low' in analysis_lower:
            return 'very_low'
        else:
            return 'unknown'
    
    def _extract_bias_assessment(self, statistical_analysis: str) -> List[str]:
        """Extract bias assessment information"""
        bias_points = []
        
        bias_keywords = [
            'bias', 'publication bias', 'selection bias', 'reporting bias',
            'performance bias', 'detection bias', 'attrition bias'
        ]
        
        lines = statistical_analysis.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in bias_keywords):
                bias_points.append(line)
        
        return bias_points[:5]  # Limit to top 5 bias points
    
    def _calculate_statistical_confidence(self, studies: List[Dict[str, Any]], statistical_analysis: str) -> float:
        """Calculate statistical confidence level"""
        if not studies:
            return 0.0
        
        # Base confidence from number of studies
        num_studies = len(studies)
        base_confidence = min(0.6, num_studies * 0.1)
        
        # Adjust based on statistical information availability
        statistical_info_score = 0.0
        for study in studies:
            abstract = study.get('abstract', '')
            if self._has_statistical_info(abstract):
                statistical_info_score += 0.1
        
        statistical_info_score = min(0.3, statistical_info_score)
        
        # Adjust based on evidence quality
        evidence_quality = self._extract_evidence_quality(statistical_analysis)
        quality_multiplier = {
            'high': 1.0,
            'moderate': 0.8,
            'low': 0.6,
            'very_low': 0.4,
            'unknown': 0.5
        }.get(evidence_quality, 0.5)
        
        total_confidence = (base_confidence + statistical_info_score) * quality_multiplier
        
        return max(0.0, min(1.0, total_confidence))
    
    def _has_statistical_info(self, abstract: str) -> bool:
        """Check if abstract contains statistical information"""
        statistical_patterns = [
            r'p\s*[<>=]\s*0\.\d+',  # p-values
            r'\d+\.?\d*\s*\([^)]*\)',  # Confidence intervals
            r'(?:odds ratio|risk ratio|hazard ratio|relative risk|effect size|cohen\'s d)',  # Effect sizes
            r'n\s*=\s*\d+'  # Sample sizes
        ]
        
        for pattern in statistical_patterns:
            if re.search(pattern, abstract, re.IGNORECASE):
                return True
        
        return False
