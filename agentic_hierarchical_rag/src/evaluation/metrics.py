"""Evaluation metrics for RAG systems."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import time
from collections import Counter
import re


@dataclass
class RAGMetrics:
    """Container for RAG evaluation metrics."""
    
    # Retrieval metrics
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    retrieval_f1: float = 0.0
    retrieval_mrr: float = 0.0  # Mean Reciprocal Rank
    retrieval_map: float = 0.0  # Mean Average Precision
    retrieval_ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    
    # Generation metrics
    answer_accuracy: float = 0.0
    answer_completeness: float = 0.0
    answer_relevance: float = 0.0
    answer_consistency: float = 0.0
    hallucination_rate: float = 0.0
    
    # Efficiency metrics
    avg_retrieval_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0
    avg_nodes_examined: float = 0.0
    avg_iterations: float = 0.0
    
    # Abstraction metrics
    level_distribution: Dict[int, float] = None
    abstraction_accuracy: float = 0.0
    navigation_efficiency: float = 0.0
    
    def __post_init__(self):
        if self.level_distribution is None:
            self.level_distribution = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "retrieval": {
                "precision": self.retrieval_precision,
                "recall": self.retrieval_recall,
                "f1": self.retrieval_f1,
                "mrr": self.retrieval_mrr,
                "map": self.retrieval_map,
                "ndcg": self.retrieval_ndcg,
            },
            "generation": {
                "accuracy": self.answer_accuracy,
                "completeness": self.answer_completeness,
                "relevance": self.answer_relevance,
                "consistency": self.answer_consistency,
                "hallucination_rate": self.hallucination_rate,
            },
            "efficiency": {
                "avg_retrieval_time_ms": self.avg_retrieval_time_ms,
                "avg_generation_time_ms": self.avg_generation_time_ms,
                "avg_total_time_ms": self.avg_total_time_ms,
                "avg_nodes_examined": self.avg_nodes_examined,
                "avg_iterations": self.avg_iterations,
            },
            "abstraction": {
                "level_distribution": self.level_distribution,
                "abstraction_accuracy": self.abstraction_accuracy,
                "navigation_efficiency": self.navigation_efficiency,
            }
        }


class MetricsCalculator:
    """Calculates various RAG metrics."""
    
    @staticmethod
    def calculate_retrieval_metrics(
        retrieved_nodes: List[str],
        relevant_nodes: List[str],
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """Calculate retrieval metrics.
        
        Args:
            retrieved_nodes: List of retrieved node IDs (ordered by relevance)
            relevant_nodes: List of actually relevant node IDs
            k: Consider only top-k retrieved nodes
            
        Returns:
            Dictionary of retrieval metrics
        """
        if k:
            retrieved_nodes = retrieved_nodes[:k]
        
        retrieved_set = set(retrieved_nodes)
        relevant_set = set(relevant_nodes)
        
        # Precision: fraction of retrieved that are relevant
        precision = 0.0
        if retrieved_set:
            precision = len(retrieved_set & relevant_set) / len(retrieved_set)
        
        # Recall: fraction of relevant that are retrieved
        recall = 0.0
        if relevant_set:
            recall = len(retrieved_set & relevant_set) / len(relevant_set)
        
        # F1 score
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, node_id in enumerate(retrieved_nodes):
            if node_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        
        # Mean Average Precision (MAP)
        ap = 0.0
        num_relevant_found = 0
        for i, node_id in enumerate(retrieved_nodes):
            if node_id in relevant_set:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / (i + 1)
                ap += precision_at_i
        
        if relevant_set:
            map_score = ap / len(relevant_set)
        else:
            map_score = 0.0
        
        # Normalized Discounted Cumulative Gain (NDCG)
        # Simplified version assuming binary relevance
        dcg = 0.0
        for i, node_id in enumerate(retrieved_nodes):
            if node_id in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because positions start at 1
        
        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), len(retrieved_nodes))))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr,
            "map": map_score,
            "ndcg": ndcg
        }
    
    @staticmethod
    def calculate_answer_metrics(
        generated_answer: str,
        reference_answer: str,
        retrieved_context: str
    ) -> Dict[str, float]:
        """Calculate answer quality metrics.
        
        Args:
            generated_answer: Generated answer
            reference_answer: Reference/ground truth answer
            retrieved_context: Retrieved context used for generation
            
        Returns:
            Dictionary of answer metrics
        """
        # Simple word overlap for accuracy (in practice, use better metrics)
        gen_words = set(generated_answer.lower().split())
        ref_words = set(reference_answer.lower().split())
        
        accuracy = len(gen_words & ref_words) / len(ref_words) if ref_words else 0.0
        
        # Completeness: how much of reference is covered
        completeness = len(gen_words & ref_words) / len(ref_words) if ref_words else 0.0
        
        # Relevance: how much of generated is relevant
        relevance = len(gen_words & ref_words) / len(gen_words) if gen_words else 0.0
        
        # Consistency: check if answer is supported by context
        context_words = set(retrieved_context.lower().split())
        supported_words = gen_words & context_words
        consistency = len(supported_words) / len(gen_words) if gen_words else 0.0
        
        # Hallucination: words in answer not in context or reference
        hallucinated_words = gen_words - context_words - ref_words
        hallucination_rate = len(hallucinated_words) / len(gen_words) if gen_words else 0.0
        
        return {
            "accuracy": accuracy,
            "completeness": completeness,
            "relevance": relevance,
            "consistency": consistency,
            "hallucination_rate": hallucination_rate
        }
    
    @staticmethod
    def calculate_efficiency_metrics(
        responses: List[Any]  # List of GenerationResponse objects
    ) -> Dict[str, float]:
        """Calculate efficiency metrics from responses.
        
        Args:
            responses: List of generation responses
            
        Returns:
            Dictionary of efficiency metrics
        """
        if not responses:
            return {
                "avg_retrieval_time_ms": 0.0,
                "avg_generation_time_ms": 0.0,
                "avg_total_time_ms": 0.0,
                "avg_nodes_examined": 0.0,
                "avg_iterations": 0.0
            }
        
        retrieval_times = []
        generation_times = []
        total_times = []
        nodes_examined = []
        iterations = []
        
        for response in responses:
            retrieval_times.append(response.retrieval_response.retrieval_time_ms)
            generation_times.append(response.generation_time_ms)
            total_times.append(response.get_total_time_ms())
            nodes_examined.append(response.retrieval_response.total_nodes_examined)
            
            # Get iterations from metadata if available
            if "iterations" in response.retrieval_response.metadata:
                iterations.append(response.retrieval_response.metadata["iterations"])
        
        return {
            "avg_retrieval_time_ms": np.mean(retrieval_times),
            "avg_generation_time_ms": np.mean(generation_times),
            "avg_total_time_ms": np.mean(total_times),
            "avg_nodes_examined": np.mean(nodes_examined),
            "avg_iterations": np.mean(iterations) if iterations else 0.0
        }
    
    @staticmethod
    def calculate_abstraction_metrics(
        responses: List[Any],  # List of GenerationResponse objects
        expected_levels: Optional[Dict[str, List[int]]] = None
    ) -> Dict[str, Any]:
        """Calculate abstraction-related metrics.
        
        Args:
            responses: List of generation responses
            expected_levels: Optional mapping of query IDs to expected levels
            
        Returns:
            Dictionary of abstraction metrics
        """
        # Collect level usage statistics
        level_counts = Counter()
        total_nodes = 0
        
        for response in responses:
            for node in response.retrieval_response.retrieved_nodes:
                level_counts[node.level] += 1
                total_nodes += 1
        
        # Calculate level distribution
        level_distribution = {}
        if total_nodes > 0:
            for level, count in level_counts.items():
                level_distribution[level] = count / total_nodes
        
        # Calculate abstraction accuracy if expected levels provided
        abstraction_accuracy = 0.0
        if expected_levels:
            correct_abstractions = 0
            total_queries = 0
            
            for response in responses:
                query_id = response.query.query_id
                if query_id in expected_levels:
                    expected = set(expected_levels[query_id])
                    actual = set(response.retrieval_response.levels_searched)
                    
                    # Check if actual levels match expected
                    if expected & actual:  # Any overlap counts as correct
                        correct_abstractions += 1
                    total_queries += 1
            
            if total_queries > 0:
                abstraction_accuracy = correct_abstractions / total_queries
        
        # Calculate navigation efficiency
        # Efficiency = 1 - (redundant_nodes / total_nodes)
        unique_nodes = set()
        redundant_count = 0
        
        for response in responses:
            for node in response.retrieval_response.retrieved_nodes:
                if node.node_id in unique_nodes:
                    redundant_count += 1
                else:
                    unique_nodes.add(node.node_id)
        
        navigation_efficiency = 1.0 - (redundant_count / total_nodes) if total_nodes > 0 else 1.0
        
        return {
            "level_distribution": level_distribution,
            "abstraction_accuracy": abstraction_accuracy,
            "navigation_efficiency": navigation_efficiency
        }
    
    @staticmethod
    def aggregate_metrics(
        retrieval_metrics: Dict[str, float],
        answer_metrics: Dict[str, float],
        efficiency_metrics: Dict[str, float],
        abstraction_metrics: Dict[str, Any]
    ) -> RAGMetrics:
        """Aggregate all metrics into RAGMetrics object.
        
        Args:
            retrieval_metrics: Retrieval metrics
            answer_metrics: Answer quality metrics
            efficiency_metrics: Efficiency metrics
            abstraction_metrics: Abstraction metrics
            
        Returns:
            RAGMetrics object
        """
        return RAGMetrics(
            # Retrieval metrics
            retrieval_precision=retrieval_metrics.get("precision", 0.0),
            retrieval_recall=retrieval_metrics.get("recall", 0.0),
            retrieval_f1=retrieval_metrics.get("f1", 0.0),
            retrieval_mrr=retrieval_metrics.get("mrr", 0.0),
            retrieval_map=retrieval_metrics.get("map", 0.0),
            retrieval_ndcg=retrieval_metrics.get("ndcg", 0.0),
            
            # Generation metrics
            answer_accuracy=answer_metrics.get("accuracy", 0.0),
            answer_completeness=answer_metrics.get("completeness", 0.0),
            answer_relevance=answer_metrics.get("relevance", 0.0),
            answer_consistency=answer_metrics.get("consistency", 0.0),
            hallucination_rate=answer_metrics.get("hallucination_rate", 0.0),
            
            # Efficiency metrics
            avg_retrieval_time_ms=efficiency_metrics.get("avg_retrieval_time_ms", 0.0),
            avg_generation_time_ms=efficiency_metrics.get("avg_generation_time_ms", 0.0),
            avg_total_time_ms=efficiency_metrics.get("avg_total_time_ms", 0.0),
            avg_nodes_examined=efficiency_metrics.get("avg_nodes_examined", 0.0),
            avg_iterations=efficiency_metrics.get("avg_iterations", 0.0),
            
            # Abstraction metrics
            level_distribution=abstraction_metrics.get("level_distribution", {}),
            abstraction_accuracy=abstraction_metrics.get("abstraction_accuracy", 0.0),
            navigation_efficiency=abstraction_metrics.get("navigation_efficiency", 0.0)
        )