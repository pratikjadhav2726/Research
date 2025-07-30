"""Demonstration of Agentic Hierarchical RAG system."""

import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ah_rag import AgenticHierarchicalRAG
from src.evaluation import BenchmarkDataset, Benchmark, Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_documents():
    """Create sample documents for demonstration."""
    documents = [
        # Document about machine learning
        """
        Machine Learning Fundamentals
        
        Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.
        
        Types of Machine Learning:
        1. Supervised Learning: The algorithm learns from labeled training data. Examples include classification and regression tasks.
        2. Unsupervised Learning: The algorithm finds patterns in unlabeled data. Examples include clustering and dimensionality reduction.
        3. Reinforcement Learning: The algorithm learns through interaction with an environment, receiving rewards or penalties.
        
        Common algorithms include decision trees, neural networks, support vector machines, and random forests. Each has its strengths and is suited for different types of problems.
        """,
        
        # Document about deep learning
        """
        Deep Learning and Neural Networks
        
        Deep learning is a specialized subset of machine learning that uses artificial neural networks with multiple layers. These networks are inspired by the human brain's structure and function.
        
        Key Concepts:
        - Neurons: Basic units that receive inputs, apply weights and biases, and produce outputs
        - Layers: Input layer, hidden layers, and output layer
        - Activation Functions: Non-linear functions like ReLU, sigmoid, and tanh
        - Backpropagation: Algorithm for training networks by propagating errors backward
        
        Popular architectures include Convolutional Neural Networks (CNNs) for image processing, Recurrent Neural Networks (RNNs) for sequential data, and Transformers for natural language processing.
        """,
        
        # Document about natural language processing
        """
        Natural Language Processing (NLP)
        
        NLP is a field at the intersection of computer science, artificial intelligence, and linguistics. It focuses on enabling computers to understand, interpret, and generate human language.
        
        Key Tasks in NLP:
        - Tokenization: Breaking text into words or subwords
        - Part-of-Speech Tagging: Identifying grammatical roles
        - Named Entity Recognition: Identifying people, places, organizations
        - Sentiment Analysis: Determining emotional tone
        - Machine Translation: Converting text between languages
        
        Modern NLP heavily relies on transformer models like BERT, GPT, and T5, which have revolutionized the field with their ability to understand context and generate coherent text.
        """,
        
        # Document about computer vision
        """
        Computer Vision Applications
        
        Computer vision enables machines to interpret and understand visual information from the world. It combines techniques from machine learning, image processing, and pattern recognition.
        
        Applications:
        - Object Detection: Identifying and locating objects in images
        - Image Classification: Categorizing images into predefined classes
        - Facial Recognition: Identifying individuals from facial features
        - Medical Imaging: Analyzing X-rays, MRIs, and CT scans
        - Autonomous Vehicles: Understanding road conditions and obstacles
        
        Convolutional Neural Networks (CNNs) are the backbone of most modern computer vision systems, with architectures like ResNet, VGG, and YOLO achieving remarkable performance.
        """
    ]
    
    return documents


def demo_basic_usage():
    """Demonstrate basic usage of AH-RAG."""
    logger.info("=== Basic Usage Demo ===")
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Initialize AH-RAG system
    ah_rag = AgenticHierarchicalRAG(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        config={
            "chunk_size": 200,
            "max_levels": 3,
            "max_iterations": 3,
        }
    )
    
    # Index documents
    logger.info("Indexing documents...")
    tree = ah_rag.index_documents(documents, show_progress=True)
    
    # Display tree statistics
    stats = ah_rag.get_tree_stats()
    logger.info(f"Tree statistics: {stats}")
    
    # Example queries
    queries = [
        "What are the main types of machine learning?",  # Factual query
        "Provide an overview of deep learning concepts",  # Thematic query
        "Compare CNNs and RNNs",  # Comparative query
        "How does NLP relate to machine learning?",  # Analytical query
    ]
    
    # Process queries
    for query in queries:
        logger.info(f"\nQuery: {query}")
        response = ah_rag.query(query)
        
        logger.info(f"Answer: {response.answer}")
        logger.info(f"Retrieved from levels: {response.retrieval_response.levels_searched}")
        logger.info(f"Nodes examined: {response.retrieval_response.total_nodes_examined}")
        logger.info(f"Total time: {response.get_total_time_ms():.2f}ms")


def demo_evaluation():
    """Demonstrate evaluation capabilities."""
    logger.info("\n=== Evaluation Demo ===")
    
    # Create synthetic benchmark dataset
    dataset = BenchmarkDataset.create_synthetic_dataset(n_queries_per_type=5)
    logger.info(f"Created benchmark dataset with {len(dataset)} queries")
    
    # Initialize system
    ah_rag = AgenticHierarchicalRAG()
    
    # Index some documents
    documents = create_sample_documents()
    ah_rag.index_documents(documents)
    
    # Run benchmark
    benchmark = Benchmark(dataset, output_dir=Path("demo_benchmark_results"))
    results = benchmark.run(ah_rag, name="demo_run", save_responses=False)
    
    logger.info(f"Benchmark results:")
    logger.info(f"  Success rate: {results['summary']['success_rate']:.2%}")
    logger.info(f"  Average response time: {results['metrics']['avg_response_time']:.2f}s")
    
    # Evaluate by query type
    evaluator = Evaluator(output_dir=Path("demo_evaluation_results"))
    
    # Note: In a real scenario, you would collect actual responses
    # and have ground truth data for evaluation


def demo_advanced_features():
    """Demonstrate advanced features."""
    logger.info("\n=== Advanced Features Demo ===")
    
    # Load configuration from file
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if config_path.exists():
        ah_rag = AgenticHierarchicalRAG.from_config(config_path)
        logger.info("Loaded configuration from file")
    else:
        ah_rag = AgenticHierarchicalRAG()
    
    # Index documents
    documents = create_sample_documents()
    tree_path = Path("demo_tree.json")
    
    # Build and save tree
    tree = ah_rag.index_documents(documents, save_tree=tree_path)
    logger.info(f"Saved tree to {tree_path}")
    
    # Load tree (simulating restart)
    ah_rag_loaded = AgenticHierarchicalRAG()
    ah_rag_loaded.load_tree(tree_path)
    logger.info("Loaded tree from disk")
    
    # Query loaded system
    response = ah_rag_loaded.query("What is backpropagation?")
    logger.info(f"Query result from loaded system: {response.answer[:100]}...")
    
    # Clean up
    if tree_path.exists():
        tree_path.unlink()


def main():
    """Run all demonstrations."""
    logger.info("Starting Agentic Hierarchical RAG demonstration\n")
    
    # Run demos
    demo_basic_usage()
    demo_evaluation()
    demo_advanced_features()
    
    logger.info("\nDemonstration complete!")


if __name__ == "__main__":
    main()