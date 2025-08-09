"""
Basic CoRS System Example

This example demonstrates how to use the CoRS (Collaborative Retrieval and Synthesis)
system to process queries with multi-agent collaboration.
"""

import asyncio
import logging
import sys
import os
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.cors_system import CoRSSystem, CoRSConfig
from core.reputation_weighted_consensus import ConsensusStrategy
from agents.retrieval_agent import ChromaDBInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_sample_knowledge_base():
    """Set up a sample knowledge base with some documents."""
    try:
        # Initialize ChromaDB
        chroma_db = ChromaDBInterface(collection_name="cors_demo")
        
        # Sample documents about renewable energy
        sample_documents = [
            {
                "id": "renewable_energy_1",
                "content": "Solar energy is a renewable energy source that converts sunlight into electricity using photovoltaic cells. Solar panels can be installed on rooftops or in large solar farms. The efficiency of modern solar panels ranges from 15% to 22%. Solar energy produces no direct carbon emissions during operation, making it environmentally friendly.",
                "metadata": {"topic": "solar_energy", "source": "energy_guide"}
            },
            {
                "id": "renewable_energy_2", 
                "content": "Wind power generates electricity using wind turbines that convert kinetic energy from moving air into electrical energy. Wind farms can be located onshore or offshore. Modern wind turbines can generate 1.5 to 3 MW of power each. Wind energy is one of the fastest-growing renewable energy sources globally.",
                "metadata": {"topic": "wind_energy", "source": "energy_guide"}
            },
            {
                "id": "renewable_energy_3",
                "content": "Studies show that renewable energy can significantly reduce electricity costs over time due to lower operational expenses compared to fossil fuels. The initial investment in renewable energy infrastructure is higher, but the long-term savings are substantial. Many countries offer tax incentives for renewable energy adoption.",
                "metadata": {"topic": "economics", "source": "economic_analysis"}
            },
            {
                "id": "renewable_energy_4",
                "content": "Renewable energy adoption faces several challenges including intermittency issues, storage costs, and grid integration complexities. However, advances in battery technology and smart grid systems are addressing many of these challenges. Energy storage solutions are becoming more affordable and efficient.",
                "metadata": {"topic": "challenges", "source": "technical_report"}
            },
            {
                "id": "renewable_energy_5",
                "content": "The environmental benefits of renewable energy include reduced greenhouse gas emissions, improved air quality, and decreased dependence on fossil fuels. Renewable energy sources like solar and wind produce no direct carbon emissions during operation. This helps combat climate change and reduces environmental pollution.",
                "metadata": {"topic": "environment", "source": "environmental_study"}
            },
            {
                "id": "ai_healthcare_1",
                "content": "Machine learning algorithms are increasingly used in medical diagnosis, particularly in medical imaging. AI systems can analyze X-rays, MRIs, and CT scans to detect abnormalities with high accuracy. Deep learning models have shown promising results in detecting cancer, fractures, and other medical conditions.",
                "metadata": {"topic": "ai_healthcare", "source": "medical_journal"}
            },
            {
                "id": "ai_healthcare_2",
                "content": "Natural language processing (NLP) is used in healthcare for processing electronic health records, clinical notes, and medical literature. NLP can extract relevant information from unstructured medical text and assist in clinical decision-making. It helps healthcare providers access and analyze patient information more efficiently.",
                "metadata": {"topic": "ai_healthcare", "source": "medical_journal"}
            },
            {
                "id": "ai_healthcare_3",
                "content": "AI in healthcare faces challenges including data privacy concerns, regulatory compliance, and the need for clinical validation. Healthcare AI systems must meet strict safety and efficacy standards before deployment. Ensuring patient data security and maintaining trust in AI systems are critical considerations.",
                "metadata": {"topic": "ai_healthcare", "source": "medical_journal"}
            }
        ]
        
        # Add documents to the knowledge base
        success = chroma_db.add_documents(sample_documents)
        if success:
            logger.info(f"Successfully added {len(sample_documents)} documents to the knowledge base")
        else:
            logger.warning("Failed to add some documents to the knowledge base")
        
        return chroma_db
        
    except Exception as e:
        logger.error(f"Failed to set up knowledge base: {e}")
        return None


async def run_cors_example():
    """Run a complete CoRS example."""
    logger.info("Starting CoRS System Example")
    
    # Set up sample knowledge base
    logger.info("Setting up sample knowledge base...")
    knowledge_base = setup_sample_knowledge_base()
    
    if not knowledge_base:
        logger.error("Failed to set up knowledge base. Exiting.")
        return
    
    # Configure CoRS system
    config = CoRSConfig(
        num_synthesizer_agents=3,
        num_retrieval_agents=1,
        consensus_strategy=ConsensusStrategy.WEIGHTED_AVERAGE,
        consensus_threshold=0.7,
        learning_rate=0.1,
        max_sub_queries=4,
        max_documents_per_query=3,
        vector_store_type="chromadb"
    )
    
    # Initialize CoRS system
    logger.info("Initializing CoRS system...")
    try:
        cors_system = CoRSSystem(config)
        logger.info("CoRS system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize CoRS system: {e}")
        return
    
    # Test queries
    test_queries = [
        "What are the benefits and challenges of renewable energy adoption?",
        "How is artificial intelligence being used in healthcare applications?",
        "What are the environmental and economic impacts of solar and wind power?"
    ]
    
    # Process each query
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Query {i}: {query}")
        logger.info(f"{'='*60}")
        
        try:
            # Process the query
            result = await cors_system.process_query(query)
            
            # Display results
            print(f"\n🔍 Query: {result['query']}")
            print(f"✅ Success: {result['success']}")
            print(f"⏱️  Processing Time: {result['processing_time']:.2f} seconds")
            print(f"📊 Status: {result['status']}")
            
            if result['success']:
                print(f"\n📝 Sub-queries generated:")
                for j, sub_query in enumerate(result['sub_queries'], 1):
                    print(f"  {j}. {sub_query}")
                
                print(f"\n🎯 Final Answer:")
                print(f"{result['answer']}")
                
                # Display metadata
                metadata = result.get('metadata', {})
                if 'session_metrics' in metadata:
                    session_metrics = metadata['session_metrics']
                    print(f"\n📈 Session Metrics:")
                    print(f"  - Sub-queries: {session_metrics.get('total_sub_queries', 0)}")
                    print(f"  - Completion rate: {session_metrics.get('completion_rate', 0):.1%}")
                    print(f"  - Total syntheses: {session_metrics.get('total_syntheses', 0)}")
                    print(f"  - Total critiques: {session_metrics.get('total_critiques', 0)}")
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"\n❌ Failed to process query: {e}")
    
    # Display system statistics
    logger.info("\n" + "="*60)
    logger.info("System Statistics")
    logger.info("="*60)
    
    try:
        stats = cors_system.get_system_stats()
        
        print(f"\n🤖 Agent Pool Statistics:")
        agent_stats = stats['agent_pool']
        print(f"  - Total agents: {agent_stats['total_agents']}")
        print(f"  - Active agents: {agent_stats['active_agents']}")
        print(f"  - Average reputation: {agent_stats['avg_reputation']:.3f}")
        print(f"  - Agent types: {agent_stats['agent_types']}")
        
        print(f"\n🏆 Top Performing Agents:")
        for agent_id, reputation in stats['top_agents'][:5]:
            print(f"  - {agent_id}: {reputation:.3f}")
        
        print(f"\n⚙️  System Configuration:")
        config_stats = stats['config']
        print(f"  - Synthesizer agents: {config_stats['num_synthesizer_agents']}")
        print(f"  - Retrieval agents: {config_stats['num_retrieval_agents']}")
        print(f"  - Consensus strategy: {config_stats['consensus_strategy']}")
        print(f"  - Consensus threshold: {config_stats['consensus_threshold']}")
        
    except Exception as e:
        logger.error(f"Error getting system statistics: {e}")
    
    logger.info("\nCoRS System Example completed successfully!")


def run_simple_sync_example():
    """Run a simple synchronous example for testing."""
    print("CoRS System - Simple Test")
    print("=" * 40)
    
    # This is a simplified version that doesn't require Redis or vector DB
    # Just to test the basic agent functionality
    
    try:
        from agents.decomposer_agent import DecomposerAgent
        from agents.base_agent import AgentConfig
        
        # Test decomposer agent
        decomposer = DecomposerAgent()
        
        test_query = "What are the benefits and challenges of renewable energy?"
        
        print(f"Testing query: {test_query}")
        result = decomposer.process({"original_query": test_query})
        
        if result["success"]:
            print(f"\nDecomposition successful!")
            print(f"Sub-queries generated:")
            for i, sub_query in enumerate(result["sub_queries"], 1):
                print(f"  {i}. {sub_query}")
        else:
            print(f"Decomposition failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"Error in simple test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("CoRS Multi-Agent RAG System Example")
    print("=" * 50)
    
    # Check if we should run the full example or simple test
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        run_simple_sync_example()
    else:
        try:
            # Run the full async example
            asyncio.run(run_cors_example())
        except KeyboardInterrupt:
            print("\nExample interrupted by user")
        except Exception as e:
            logger.error(f"Example failed: {e}")
            import traceback
            traceback.print_exc()
            
            print("\nIf you're having issues with dependencies, try running with --simple flag:")
            print("python basic_example.py --simple")