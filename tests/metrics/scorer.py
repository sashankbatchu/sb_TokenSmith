from typing import Dict, List, Any, Optional
from tests.metrics.registry import MetricRegistry

NON_AGGREGATE_METRICS = {"chunk_retrieval"}


class SimilarityScorer:
    """Main scorer that uses selected metrics."""
    
    def __init__(self, enabled_metrics: Optional[List[str]] = None):
        self.registry = MetricRegistry()
        self.enabled_metrics = enabled_metrics or ["all"]
    
    def _get_active_metrics(self) -> Dict[str, Any]:
        """Get metrics that should be used for scoring."""
        available = self.registry.get_available_metrics()
        
        if "all" in self.enabled_metrics:
            return available
        
        active = {}
        for name in self.enabled_metrics:
            if name in available:
                active[name] = available[name]
            else:
                print(f"ERROR: Metric '{name}' not available")
        
        return active
    
    def calculate_scores(self, answer: str, expected: str, keywords: Optional[List[str]] = None, question: Optional[str] = None, ideal_retrieved_chunks: Optional[List[int]] = None, actual_retrieved_chunks: Optional[List[int]] = None) -> Dict[str, Any]:
        """Calculate scores using active metrics."""
        active_metrics = self._get_active_metrics()
        
        if not active_metrics:
            return {"error": "No metrics available"}
        
        scores = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for name, metric in active_metrics.items():
            # For LLM judge metrics, pass question instead of expected answer
            if name in ("llm_judge", "async_llm_judge") and question:
                score = metric.calculate(answer, question, keywords)
            elif name == "chunk_retrieval":
                score = metric.calculate(ideal_retrieved_chunks, actual_retrieved_chunks)
            else:
                score = metric.calculate(answer, expected, keywords)
            scores[f"{name}_similarity"] = score
            
            weight = metric.weight
            # Some metrics are useful diagnostics but should not affect benchmark pass/fail.
            if weight > 0 and name not in NON_AGGREGATE_METRICS:
                total_weighted_score += score * weight
                total_weight += weight
        
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        if keywords:
            answer_lower = answer.lower()
            keywords_matched = sum(1 for kw in keywords if kw.lower() in answer_lower)
        else:
            keywords_matched = 0
        
        return {
            **scores,
            "final_score": final_score,
            "keywords_matched": keywords_matched,
            "active_metrics": list(active_metrics.keys()),
            "non_aggregate_metrics": sorted(name for name in active_metrics if name in NON_AGGREGATE_METRICS),
        }
