"""Main evaluator for Agentic Hierarchical RAG."""

from typing import List, Dict, Any, Optional, Callable
import logging
from pathlib import Path
import json
import time
from datetime import datetime

from ..core import GenerationResponse
from .metrics import RAGMetrics, MetricsCalculator
from .benchmark import BenchmarkDataset, Benchmark

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates AH-RAG system performance."""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        llm_judge: Optional[Callable[[str, str], Dict[str, float]]] = None
    ):
        """Initialize evaluator.
        
        Args:
            output_dir: Directory for saving evaluation results
            llm_judge: Optional LLM-based evaluation function
        """
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_judge = llm_judge
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_responses(
        self,
        responses: List[GenerationResponse],
        ground_truth: Optional[Dict[str, Any]] = None,
        save_results: bool = True
    ) -> RAGMetrics:
        """Evaluate a list of responses.
        
        Args:
            responses: List of generation responses to evaluate
            ground_truth: Optional ground truth data
            save_results: Whether to save evaluation results
            
        Returns:
            Aggregated RAGMetrics
        """
        logger.info(f"Evaluating {len(responses)} responses")
        
        # Calculate efficiency metrics
        efficiency_metrics = self.metrics_calculator.calculate_efficiency_metrics(responses)
        
        # Calculate abstraction metrics
        expected_levels = None
        if ground_truth and "expected_levels" in ground_truth:
            expected_levels = ground_truth["expected_levels"]
        
        abstraction_metrics = self.metrics_calculator.calculate_abstraction_metrics(
            responses, expected_levels
        )
        
        # Initialize aggregated metrics for retrieval and generation
        all_retrieval_metrics = []
        all_answer_metrics = []
        
        # Evaluate each response
        for i, response in enumerate(responses):
            # Get ground truth for this query if available
            query_id = response.query.query_id
            
            # Retrieval metrics
            if ground_truth and "relevant_nodes" in ground_truth:
                relevant = ground_truth["relevant_nodes"].get(query_id, [])
                retrieved = [n.node_id for n in response.retrieval_response.retrieved_nodes]
                
                retrieval_metrics = self.metrics_calculator.calculate_retrieval_metrics(
                    retrieved, relevant
                )
                all_retrieval_metrics.append(retrieval_metrics)
            
            # Answer metrics
            if ground_truth and "expected_answers" in ground_truth:
                expected = ground_truth["expected_answers"].get(query_id, "")
                context = response.retrieval_response.get_context()
                
                answer_metrics = self.metrics_calculator.calculate_answer_metrics(
                    response.answer, expected, context
                )
                
                # Use LLM judge if available for more sophisticated evaluation
                if self.llm_judge:
                    llm_scores = self.llm_judge(response.answer, expected)
                    answer_metrics.update(llm_scores)
                
                all_answer_metrics.append(answer_metrics)
        
        # Aggregate metrics
        aggregated_retrieval = self._aggregate_metric_list(all_retrieval_metrics)
        aggregated_answer = self._aggregate_metric_list(all_answer_metrics)
        
        # Create final metrics
        metrics = self.metrics_calculator.aggregate_metrics(
            aggregated_retrieval,
            aggregated_answer,
            efficiency_metrics,
            abstraction_metrics
        )
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(metrics, responses, ground_truth)
        
        return metrics
    
    def evaluate_query_types(
        self,
        responses: List[GenerationResponse],
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Dict[str, RAGMetrics]:
        """Evaluate responses grouped by query type.
        
        Args:
            responses: List of responses to evaluate
            ground_truth: Optional ground truth data
            
        Returns:
            Dictionary mapping query type to metrics
        """
        from ..core import QueryType
        
        # Group responses by query type
        responses_by_type = {}
        for response in responses:
            query_type = response.query.query_type
            if query_type not in responses_by_type:
                responses_by_type[query_type] = []
            responses_by_type[query_type].append(response)
        
        # Evaluate each type separately
        metrics_by_type = {}
        for query_type, type_responses in responses_by_type.items():
            logger.info(f"Evaluating {len(type_responses)} {query_type.value} queries")
            
            # Filter ground truth for this type if available
            type_ground_truth = None
            if ground_truth:
                type_ground_truth = self._filter_ground_truth_by_type(
                    ground_truth, type_responses
                )
            
            metrics = self.evaluate_responses(
                type_responses, 
                type_ground_truth,
                save_results=False
            )
            metrics_by_type[query_type.value] = metrics
        
        return metrics_by_type
    
    def run_ablation_study(
        self,
        base_system: Any,  # AgenticHierarchicalRAG
        ablation_configs: Dict[str, Dict[str, Any]],
        benchmark_dataset: BenchmarkDataset
    ) -> Dict[str, Any]:
        """Run ablation study on different system configurations.
        
        Args:
            base_system: Base AH-RAG system
            ablation_configs: Dictionary of configuration variations
            benchmark_dataset: Dataset to test on
            
        Returns:
            Results of ablation study
        """
        logger.info(f"Running ablation study with {len(ablation_configs)} configurations")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "dataset": benchmark_dataset.name,
            "configurations": {},
            "comparison": {}
        }
        
        # Run benchmark for each configuration
        benchmark = Benchmark(benchmark_dataset, self.output_dir / "ablation")
        
        for config_name, config_params in ablation_configs.items():
            logger.info(f"Testing configuration: {config_name}")
            
            # Apply configuration (this is simplified - actual implementation
            # would modify the system based on config_params)
            modified_system = self._apply_configuration(base_system, config_params)
            
            # Run benchmark
            bench_results = benchmark.run(
                modified_system,
                name=f"ablation_{config_name}",
                save_responses=False
            )
            
            results["configurations"][config_name] = {
                "params": config_params,
                "metrics": bench_results["metrics"],
                "summary": bench_results["summary"]
            }
        
        # Compare configurations
        results["comparison"] = self._compare_ablation_results(results["configurations"])
        
        # Save ablation study results
        results_path = self.output_dir / "ablation_study_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved ablation study results to {results_path}")
        
        return results
    
    def _aggregate_metric_list(
        self,
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate a list of metric dictionaries.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        if not metrics_list:
            return {}
        
        aggregated = {}
        keys = metrics_list[0].keys()
        
        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)
        
        return aggregated
    
    def _filter_ground_truth_by_type(
        self,
        ground_truth: Dict[str, Any],
        responses: List[GenerationResponse]
    ) -> Dict[str, Any]:
        """Filter ground truth data for specific responses.
        
        Args:
            ground_truth: Full ground truth data
            responses: Responses to filter for
            
        Returns:
            Filtered ground truth
        """
        query_ids = {r.query.query_id for r in responses}
        
        filtered = {}
        
        if "relevant_nodes" in ground_truth:
            filtered["relevant_nodes"] = {
                qid: nodes for qid, nodes in ground_truth["relevant_nodes"].items()
                if qid in query_ids
            }
        
        if "expected_answers" in ground_truth:
            filtered["expected_answers"] = {
                qid: ans for qid, ans in ground_truth["expected_answers"].items()
                if qid in query_ids
            }
        
        if "expected_levels" in ground_truth:
            filtered["expected_levels"] = {
                qid: levels for qid, levels in ground_truth["expected_levels"].items()
                if qid in query_ids
            }
        
        return filtered
    
    def _save_evaluation_results(
        self,
        metrics: RAGMetrics,
        responses: List[GenerationResponse],
        ground_truth: Optional[Dict[str, Any]]
    ) -> None:
        """Save evaluation results to disk.
        
        Args:
            metrics: Calculated metrics
            responses: Evaluated responses
            ground_truth: Ground truth data used
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_path = self.output_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        # Save detailed results
        detailed_results = {
            "timestamp": timestamp,
            "n_responses": len(responses),
            "metrics": metrics.to_dict(),
            "has_ground_truth": ground_truth is not None,
            "query_types": {}
        }
        
        # Group by query type
        from collections import Counter
        query_types = Counter(r.query.query_type.value for r in responses)
        detailed_results["query_types"] = dict(query_types)
        
        results_path = self.output_dir / f"evaluation_results_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(detailed_results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {self.output_dir}")
    
    def _apply_configuration(
        self,
        base_system: Any,
        config_params: Dict[str, Any]
    ) -> Any:
        """Apply configuration parameters to system.
        
        This is a placeholder - actual implementation would modify
        the system based on config_params.
        
        Args:
            base_system: Base system to modify
            config_params: Configuration parameters
            
        Returns:
            Modified system
        """
        # In a real implementation, this would:
        # - Disable certain components (e.g., self-reflection)
        # - Change parameters (e.g., max_iterations)
        # - Switch strategies (e.g., retrieval method)
        
        return base_system
    
    def _compare_ablation_results(
        self,
        configurations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare results across ablation configurations.
        
        Args:
            configurations: Results for each configuration
            
        Returns:
            Comparison analysis
        """
        comparison = {
            "best_by_metric": {},
            "relative_performance": {}
        }
        
        # Find best configuration for each metric
        metrics_to_compare = [
            "success_rate", "avg_response_time", "avg_nodes_retrieved"
        ]
        
        for metric in metrics_to_compare:
            best_config = None
            best_value = None
            
            for config_name, config_data in configurations.items():
                value = config_data["metrics"].get(metric)
                if value is not None:
                    if best_value is None or (
                        (metric == "avg_response_time" and value < best_value) or
                        (metric != "avg_response_time" and value > best_value)
                    ):
                        best_value = value
                        best_config = config_name
            
            comparison["best_by_metric"][metric] = {
                "config": best_config,
                "value": best_value
            }
        
        # Calculate relative performance
        if "baseline" in configurations:
            baseline_metrics = configurations["baseline"]["metrics"]
            
            for config_name, config_data in configurations.items():
                if config_name == "baseline":
                    continue
                
                relative = {}
                for metric in metrics_to_compare:
                    baseline_val = baseline_metrics.get(metric, 1)
                    config_val = config_data["metrics"].get(metric, 1)
                    
                    if baseline_val > 0:
                        relative[metric] = (config_val - baseline_val) / baseline_val
                
                comparison["relative_performance"][config_name] = relative
        
        return comparison