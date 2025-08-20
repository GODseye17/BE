"""
Relevance Feedback Tracking System
"""
import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class RelevanceTracker:
    """Track query results and relevance scores for feedback-based optimization"""
    
    def __init__(self, feedback_file: str = "feedback/relevance_feedback.json"):
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing feedback data
        self.feedback_data = self._load_feedback_data()
        
        # Default relevance thresholds
        self.default_thresholds = {
            'min_score': 0.4,
            'high_relevance': 0.7,
            'medium_relevance': 0.5,
            'low_relevance': 0.3
        }
        
        # Current adaptive thresholds
        self.current_thresholds = self.default_thresholds.copy()
        
        # Track query patterns
        self.query_patterns = {}
    
    def _load_feedback_data(self) -> Dict[str, Any]:
        """Load feedback data from file"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"✅ Loaded feedback data with {len(data.get('queries', []))} queries")
                return data
            except Exception as e:
                logger.warning(f"⚠️ Could not load feedback data: {e}")
        
        # Initialize new feedback data structure
        return {
            'queries': [],
            'articles': {},
            'threshold_adjustments': [],
            'query_patterns': {},
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_feedback_data(self):
        """Save feedback data to file"""
        try:
            self.feedback_data['last_updated'] = datetime.now().isoformat()
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
            logger.debug("✅ Feedback data saved")
        except Exception as e:
            logger.error(f"❌ Error saving feedback data: {e}")
    
    def record_query_result(self, query: str, topic_id: str, articles: List[Dict[str, Any]], 
                          relevance_scores: List[float], query_type: str = "general"):
        """Record query results and relevance scores"""
        query_record = {
            'query': query,
            'topic_id': topic_id,
            'query_type': query_type,
            'timestamp': datetime.now().isoformat(),
            'article_count': len(articles),
            'avg_relevance_score': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
            'min_relevance_score': min(relevance_scores) if relevance_scores else 0.0,
            'max_relevance_score': max(relevance_scores) if relevance_scores else 0.0,
            'articles': [
                {
                    'pmid': article.get('pubmed_id', 'unknown'),
                    'title': article.get('title', ''),
                    'relevance_score': score,
                    'feedback_given': False
                }
                for article, score in zip(articles, relevance_scores)
            ]
        }
        
        self.feedback_data['queries'].append(query_record)
        
        # Update query patterns
        self._update_query_patterns(query, query_type, len(articles), 
                                  query_record['avg_relevance_score'])
        
        self._save_feedback_data()
        logger.info(f"📊 Recorded query result: {query} ({len(articles)} articles)")
    
    def record_article_feedback(self, query: str, pmid: str, is_relevant: bool, 
                              user_score: Optional[float] = None):
        """Record user feedback for a specific article"""
        # Find the query record
        query_record = None
        for qr in self.feedback_data['queries']:
            if qr['query'] == query:
                query_record = qr
                break
        
        if not query_record:
            logger.warning(f"⚠️ Query not found for feedback: {query}")
            return
        
        # Update article feedback
        for article in query_record['articles']:
            if article['pmid'] == pmid:
                article['feedback_given'] = True
                article['user_relevant'] = is_relevant
                article['user_score'] = user_score
                article['feedback_timestamp'] = datetime.now().isoformat()
                break
        
        # Update article-specific data
        if pmid not in self.feedback_data['articles']:
            self.feedback_data['articles'][pmid] = {
                'feedback_count': 0,
                'relevant_count': 0,
                'avg_user_score': 0.0,
                'feedback_history': []
            }
        
        article_data = self.feedback_data['articles'][pmid]
        article_data['feedback_count'] += 1
        if is_relevant:
            article_data['relevant_count'] += 1
        
        if user_score is not None:
            current_avg = article_data['avg_user_score']
            feedback_count = article_data['feedback_count']
            article_data['avg_user_score'] = (current_avg * (feedback_count - 1) + user_score) / feedback_count
        
        # Record feedback history
        article_data['feedback_history'].append({
            'query': query,
            'is_relevant': is_relevant,
            'user_score': user_score,
            'timestamp': datetime.now().isoformat()
        })
        
        self._save_feedback_data()
        logger.info(f"✅ Recorded feedback for PMID {pmid}: {'relevant' if is_relevant else 'not relevant'}")
    
    def record_query_satisfaction(self, query: str, satisfaction_score: float, 
                                feedback_text: Optional[str] = None):
        """Record overall query satisfaction score"""
        satisfaction_record = {
            'query': query,
            'satisfaction_score': satisfaction_score,
            'feedback_text': feedback_text,
            'timestamp': datetime.now().isoformat()
        }
        
        if 'query_satisfaction' not in self.feedback_data:
            self.feedback_data['query_satisfaction'] = []
        
        self.feedback_data['query_satisfaction'].append(satisfaction_record)
        
        # Adjust thresholds based on satisfaction
        self._adjust_thresholds_based_on_satisfaction(satisfaction_score)
        
        self._save_feedback_data()
        logger.info(f"✅ Recorded query satisfaction: {satisfaction_score}/5.0")
    
    def _update_query_patterns(self, query: str, query_type: str, article_count: int, 
                             avg_relevance: float):
        """Update query pattern analysis"""
        if 'query_patterns' not in self.feedback_data:
            self.feedback_data['query_patterns'] = {}
        
        patterns = self.feedback_data['query_patterns']
        
        if query_type not in patterns:
            patterns[query_type] = {
                'query_count': 0,
                'avg_article_count': 0.0,
                'avg_relevance_score': 0.0,
                'queries': []
            }
        
        pattern_data = patterns[query_type]
        pattern_data['query_count'] += 1
        
        # Update running averages
        current_count = pattern_data['query_count']
        pattern_data['avg_article_count'] = (
            (pattern_data['avg_article_count'] * (current_count - 1) + article_count) / current_count
        )
        pattern_data['avg_relevance_score'] = (
            (pattern_data['avg_relevance_score'] * (current_count - 1) + avg_relevance) / current_count
        )
        
        pattern_data['queries'].append({
            'query': query,
            'article_count': article_count,
            'avg_relevance': avg_relevance,
            'timestamp': datetime.now().isoformat()
        })
    
    def _adjust_thresholds_based_on_satisfaction(self, satisfaction_score: float):
        """Dynamically adjust relevance thresholds based on user satisfaction"""
        adjustment_record = {
            'timestamp': datetime.now().isoformat(),
            'previous_thresholds': self.current_thresholds.copy(),
            'satisfaction_score': satisfaction_score,
            'reason': 'user_satisfaction'
        }
        
        # Adjust thresholds based on satisfaction
        if satisfaction_score < 2.0:  # Low satisfaction
            # Lower thresholds to get more results
            self.current_thresholds['min_score'] = max(0.2, self.current_thresholds['min_score'] - 0.1)
            self.current_thresholds['high_relevance'] = max(0.5, self.current_thresholds['high_relevance'] - 0.1)
            adjustment_record['adjustment'] = 'lowered_thresholds'
            
        elif satisfaction_score > 4.0:  # High satisfaction
            # Raise thresholds for better precision
            self.current_thresholds['min_score'] = min(0.6, self.current_thresholds['min_score'] + 0.05)
            self.current_thresholds['high_relevance'] = min(0.9, self.current_thresholds['high_relevance'] + 0.05)
            adjustment_record['adjustment'] = 'raised_thresholds'
        
        adjustment_record['new_thresholds'] = self.current_thresholds.copy()
        
        if 'threshold_adjustments' not in self.feedback_data:
            self.feedback_data['threshold_adjustments'] = []
        
        self.feedback_data['threshold_adjustments'].append(adjustment_record)
        
        logger.info(f"🔧 Adjusted thresholds based on satisfaction {satisfaction_score}: {adjustment_record['adjustment']}")
    
    def get_adaptive_thresholds(self, query_type: str = "general") -> Dict[str, float]:
        """Get adaptive relevance thresholds based on feedback"""
        # Check if we have enough data for this query type
        patterns = self.feedback_data.get('query_patterns', {})
        if query_type in patterns:
            pattern_data = patterns[query_type]
            if pattern_data['query_count'] >= 5:  # Need at least 5 queries for pattern
                avg_relevance = pattern_data['avg_relevance_score']
                
                # Adjust thresholds based on historical performance
                if avg_relevance < 0.4:
                    # Low historical relevance, lower thresholds
                    return {
                        'min_score': max(0.2, self.current_thresholds['min_score'] - 0.1),
                        'high_relevance': max(0.5, self.current_thresholds['high_relevance'] - 0.1),
                        'medium_relevance': max(0.3, self.current_thresholds['medium_relevance'] - 0.1),
                        'low_relevance': max(0.1, self.current_thresholds['low_relevance'] - 0.1)
                    }
                elif avg_relevance > 0.6:
                    # High historical relevance, raise thresholds
                    return {
                        'min_score': min(0.6, self.current_thresholds['min_score'] + 0.05),
                        'high_relevance': min(0.9, self.current_thresholds['high_relevance'] + 0.05),
                        'medium_relevance': min(0.7, self.current_thresholds['medium_relevance'] + 0.05),
                        'low_relevance': min(0.5, self.current_thresholds['low_relevance'] + 0.05)
                    }
        
        # Return current thresholds if no pattern data
        return self.current_thresholds.copy()
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of feedback data"""
        queries = self.feedback_data.get('queries', [])
        articles = self.feedback_data.get('articles', {})
        satisfaction = self.feedback_data.get('query_satisfaction', [])
        
        # Calculate feedback statistics
        total_queries = len(queries)
        total_articles = len(articles)
        total_feedback = sum(1 for q in queries for a in q.get('articles', []) if a.get('feedback_given'))
        
        avg_satisfaction = 0.0
        if satisfaction:
            avg_satisfaction = sum(s['satisfaction_score'] for s in satisfaction) / len(satisfaction)
        
        return {
            'total_queries': total_queries,
            'total_articles': total_articles,
            'total_feedback_given': total_feedback,
            'average_satisfaction': avg_satisfaction,
            'current_thresholds': self.current_thresholds,
            'query_patterns': self.feedback_data.get('query_patterns', {}),
            'last_updated': self.feedback_data.get('last_updated', '')
        }
    
    def reset_thresholds(self):
        """Reset thresholds to default values"""
        self.current_thresholds = self.default_thresholds.copy()
        logger.info("🔄 Reset relevance thresholds to default values")
