"""Benchmarking framework for Agentic Hierarchical RAG."""

from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import logging
from tqdm import tqdm
import pandas as pd

from ..core import QueryType

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkQuery:
    """A single benchmark query with ground truth."""
    
    query_id: str
    query_text: str
    query_type: QueryType
    expected_answer: str
    relevant_node_ids: List[str]
    expected_levels: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "query_type": self.query_type.value,
            "expected_answer": self.expected_answer,
            "relevant_node_ids": self.relevant_node_ids,
            "expected_levels": self.expected_levels,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkQuery":
        """Create from dictionary."""
        return cls(
            query_id=data["query_id"],
            query_text=data["query_text"],
            query_type=QueryType(data["query_type"]),
            expected_answer=data["expected_answer"],
            relevant_node_ids=data["relevant_node_ids"],
            expected_levels=data["expected_levels"],
            metadata=data.get("metadata", {})
        )


@dataclass
class BenchmarkDataset:
    """A collection of benchmark queries."""
    
    name: str
    description: str
    queries: List[BenchmarkQuery]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Get number of queries."""
        return len(self.queries)
    
    def get_by_type(self, query_type: QueryType) -> List[BenchmarkQuery]:
        """Get queries of a specific type."""
        return [q for q in self.queries if q.query_type == query_type]
    
    def save(self, path: Path) -> None:
        """Save dataset to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "name": self.name,
            "description": self.description,
            "queries": [q.to_dict() for q in self.queries],
            "metadata": self.metadata
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved benchmark dataset to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "BenchmarkDataset":
        """Load dataset from JSON file."""
        path = Path(path)
        
        with open(path, "r") as f:
            data = json.load(f)
        
        queries = [BenchmarkQuery.from_dict(q) for q in data["queries"]]
        
        return cls(
            name=data["name"],
            description=data["description"],
            queries=queries,
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def create_synthetic_dataset(
        cls,
        name: str = "Synthetic AH-RAG Benchmark",
        n_queries_per_type: int = 20
    ) -> "BenchmarkDataset":
        """Create a synthetic benchmark dataset.
        
        Args:
            name: Dataset name
            n_queries_per_type: Number of queries per query type
            
        Returns:
            Synthetic benchmark dataset
        """
        queries = []
        query_id = 0
        
        # Factual queries (need leaf-level detail)
        factual_templates = [
            "What is the exact date of {}?",
            "How many {} are mentioned in the document?",
            "What is the specific formula for {}?",
            "Who is the author of {}?",
            "What is the definition of {}?"
        ]
        
        for i in range(n_queries_per_type):
            template = factual_templates[i % len(factual_templates)]
            query = BenchmarkQuery(
                query_id=f"Q{query_id:04d}",
                query_text=template.format(f"concept_{i}"),
                query_type=QueryType.FACTUAL,
                expected_answer=f"The answer is detail_{i}",
                relevant_node_ids=[f"node_leaf_{i}"],
                expected_levels=[0],
                metadata={"synthetic": True}
            )
            queries.append(query)
            query_id += 1
        
        # Thematic queries (need high-level summaries)
        thematic_templates = [
            "What are the main themes in {}?",
            "Provide an overview of {}",
            "Summarize the key concepts related to {}",
            "What is the general approach to {}?",
            "Describe the overall significance of {}"
        ]
        
        for i in range(n_queries_per_type):
            template = thematic_templates[i % len(thematic_templates)]
            query = BenchmarkQuery(
                query_id=f"Q{query_id:04d}",
                query_text=template.format(f"topic_{i}"),
                query_type=QueryType.THEMATIC,
                expected_answer=f"The main themes are theme_{i}",
                relevant_node_ids=[f"node_high_{i}"],
                expected_levels=[2, 3],
                metadata={"synthetic": True}
            )
            queries.append(query)
            query_id += 1
        
        # Comparative queries (need multiple perspectives)
        comparative_templates = [
            "Compare {} and {}",
            "What are the differences between {} and {}?",
            "Which is better: {} or {}?",
            "Contrast the approaches of {} versus {}",
            "What are the pros and cons of {} compared to {}?"
        ]
        
        for i in range(n_queries_per_type):
            template = comparative_templates[i % len(comparative_templates)]
            query = BenchmarkQuery(
                query_id=f"Q{query_id:04d}",
                query_text=template.format(f"option_A_{i}", f"option_B_{i}"),
                query_type=QueryType.COMPARATIVE,
                expected_answer=f"Comparison shows difference_{i}",
                relevant_node_ids=[f"node_mid_{i}", f"node_mid_{i+1}"],
                expected_levels=[1, 2],
                metadata={"synthetic": True}
            )
            queries.append(query)
            query_id += 1
        
        # Analytical queries (need multi-level reasoning)
        analytical_templates = [
            "Analyze the impact of {} on {}",
            "Explain why {} leads to {}",
            "What is the relationship between {} and {}?",
            "How does {} influence {}?",
            "Evaluate the effectiveness of {} for {}"
        ]
        
        for i in range(n_queries_per_type):
            template = analytical_templates[i % len(analytical_templates)]
            query = BenchmarkQuery(
                query_id=f"Q{query_id:04d}",
                query_text=template.format(f"cause_{i}", f"effect_{i}"),
                query_type=QueryType.ANALYTICAL,
                expected_answer=f"Analysis shows relationship_{i}",
                relevant_node_ids=[f"node_leaf_{i}", f"node_mid_{i}", f"node_high_{i}"],
                expected_levels=[0, 1, 2],
                metadata={"synthetic": True}
            )
            queries.append(query)
            query_id += 1
        
        return cls(
            name=name,
            description="Synthetic benchmark dataset for testing AH-RAG",
            queries=queries,
            metadata={
                "synthetic": True,
                "n_queries": len(queries),
                "n_per_type": n_queries_per_type
            }
        )


class Benchmark:
    """Runs benchmarks and collects results."""
    
    def __init__(
        self,
        dataset: BenchmarkDataset,
        output_dir: Optional[Path] = None
    ):
        """Initialize benchmark runner.
        
        Args:
            dataset: Benchmark dataset to use
            output_dir: Directory to save results
        """
        self.dataset = dataset
        self.output_dir = Path(output_dir) if output_dir else Path("benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        rag_system: Any,  # AgenticHierarchicalRAG instance
        name: str = "benchmark_run",
        save_responses: bool = True,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """Run benchmark on RAG system.
        
        Args:
            rag_system: RAG system to benchmark
            name: Name for this benchmark run
            save_responses: Whether to save individual responses
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Starting benchmark '{name}' with {len(self.dataset)} queries")
        
        results = {
            "name": name,
            "dataset": self.dataset.name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "queries": [],
            "responses": [],
            "metrics": {},
            "summary": {}
        }
        
        # Run queries
        for query in tqdm(self.dataset.queries, desc="Running queries", disable=not progress_bar):
            start_time = time.time()
            
            try:
                # Process query
                response = rag_system.query(query.query_text)
                
                # Record result
                result = {
                    "query_id": query.query_id,
                    "query_type": query.query_type.value,
                    "success": True,
                    "response_time": time.time() - start_time,
                    "answer": response.answer,
                    "retrieved_nodes": [n.node_id for n in response.retrieval_response.retrieved_nodes],
                    "levels_searched": response.retrieval_response.levels_searched,
                    "error": None
                }
                
                if save_responses:
                    results["responses"].append(response)
                
            except Exception as e:
                logger.error(f"Error processing query {query.query_id}: {e}")
                result = {
                    "query_id": query.query_id,
                    "query_type": query.query_type.value,
                    "success": False,
                    "response_time": time.time() - start_time,
                    "answer": None,
                    "retrieved_nodes": [],
                    "levels_searched": [],
                    "error": str(e)
                }
            
            results["queries"].append(result)
        
        # Calculate metrics
        results["metrics"] = self._calculate_metrics(results["queries"], results["responses"])
        results["summary"] = self._create_summary(results)
        
        # Save results
        self._save_results(results, name)
        
        return results
    
    def compare_systems(
        self,
        systems: Dict[str, Any],  # name -> RAG system
        save_comparison: bool = True
    ) -> pd.DataFrame:
        """Compare multiple RAG systems.
        
        Args:
            systems: Dictionary mapping system names to RAG instances
            save_comparison: Whether to save comparison results
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(systems)} systems")
        
        all_results = {}
        
        # Run benchmark for each system
        for name, system in systems.items():
            logger.info(f"Benchmarking system: {name}")
            results = self.run(system, name=name, save_responses=False)
            all_results[name] = results
        
        # Create comparison DataFrame
        comparison_data = []
        
        for name, results in all_results.items():
            metrics = results["metrics"]
            row = {
                "system": name,
                "retrieval_f1": metrics.get("retrieval_f1", 0),
                "answer_accuracy": metrics.get("answer_accuracy", 0),
                "avg_total_time_ms": metrics.get("avg_total_time_ms", 0),
                "abstraction_accuracy": metrics.get("abstraction_accuracy", 0),
                "success_rate": results["summary"]["success_rate"]
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        if save_comparison:
            comparison_path = self.output_dir / "system_comparison.csv"
            df.to_csv(comparison_path, index=False)
            logger.info(f"Saved comparison to {comparison_path}")
        
        return df
    
    def _calculate_metrics(
        self,
        query_results: List[Dict[str, Any]],
        responses: List[Any]
    ) -> Dict[str, float]:
        """Calculate aggregate metrics from results.
        
        Args:
            query_results: List of query result dictionaries
            responses: List of response objects
            
        Returns:
            Dictionary of metrics
        """
        from .metrics import MetricsCalculator
        
        # Basic success metrics
        successful = [r for r in query_results if r["success"]]
        success_rate = len(successful) / len(query_results) if query_results else 0
        
        # Average response time
        avg_response_time = sum(r["response_time"] for r in query_results) / len(query_results)
        
        # Retrieval metrics (simplified - would need ground truth)
        avg_nodes_retrieved = sum(len(r["retrieved_nodes"]) for r in successful) / len(successful) if successful else 0
        
        # Level usage
        level_counts = {}
        for result in successful:
            for level in result["levels_searched"]:
                level_counts[level] = level_counts.get(level, 0) + 1
        
        metrics = {
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "avg_nodes_retrieved": avg_nodes_retrieved,
            "level_usage": level_counts
        }
        
        # If we have responses, calculate more detailed metrics
        if responses:
            # This would require ground truth data
            pass
        
        return metrics
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics.
        
        Args:
            results: Benchmark results
            
        Returns:
            Summary dictionary
        """
        query_results = results["queries"]
        
        # Group by query type
        by_type = {}
        for qt in QueryType:
            type_results = [r for r in query_results if r["query_type"] == qt.value]
            if type_results:
                successful = [r for r in type_results if r["success"]]
                by_type[qt.value] = {
                    "total": len(type_results),
                    "successful": len(successful),
                    "success_rate": len(successful) / len(type_results),
                    "avg_time": sum(r["response_time"] for r in type_results) / len(type_results)
                }
        
        return {
            "total_queries": len(query_results),
            "successful_queries": len([r for r in query_results if r["success"]]),
            "success_rate": results["metrics"]["success_rate"],
            "by_query_type": by_type
        }
    
    def _save_results(self, results: Dict[str, Any], name: str) -> None:
        """Save benchmark results.
        
        Args:
            results: Results to save
            name: Name for the results file
        """
        # Save main results (without response objects)
        results_copy = results.copy()
        results_copy.pop("responses", None)  # Remove response objects
        
        results_path = self.output_dir / f"{name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results_copy, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
        
        # Save summary as CSV for easy viewing
        summary_data = []
        for query_result in results["queries"]:
            summary_data.append({
                "query_id": query_result["query_id"],
                "query_type": query_result["query_type"],
                "success": query_result["success"],
                "response_time": query_result["response_time"],
                "nodes_retrieved": len(query_result["retrieved_nodes"]),
                "levels_searched": len(query_result["levels_searched"])
            })
        
        df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / f"{name}_summary.csv"
        df.to_csv(summary_path, index=False)
        
        logger.info(f"Saved summary to {summary_path}")