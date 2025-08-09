"""
CoRS Collaborative Cache Demonstration

This example demonstrates the true power of the CoRS architecture:
- Agents share a dynamic workspace that acts as an intelligent cache
- Information found by one agent is immediately available to others
- Verification counts build consensus automatically
- Massive efficiency gains through knowledge reuse

The demo shows:
1. Traditional multi-agent RAG (each agent queries independently)
2. CoRS collaborative approach (agents share through workspace)
3. Performance comparison and efficiency metrics
"""

import asyncio
import time
import logging
from typing import List, Dict, Any
import redis
from langchain_openai import ChatOpenAI

# Mock implementations for demo purposes
class MockVectorStore:
    """Mock vector store that simulates expensive knowledge base queries"""
    
    def __init__(self, name: str = "MockKnowledgeBase"):
        self.name = name
        self.query_count = 0
        self.query_delay = 1.0  # Simulate expensive queries
        
        # Mock knowledge base with sample documents
        self.documents = {
            "artificial intelligence": [
                "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence.",
                "Machine learning is a subset of AI that enables systems to learn from data without explicit programming.",
                "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
                "AI applications include natural language processing, computer vision, and robotics."
            ],
            "machine learning": [
                "Machine learning algorithms can be supervised, unsupervised, or reinforcement-based.",
                "Supervised learning uses labeled training data to make predictions on new data.",
                "Unsupervised learning finds hidden patterns in data without labeled examples.",
                "Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn."
            ],
            "neural networks": [
                "Neural networks are computing systems inspired by biological neural networks.",
                "They consist of interconnected nodes (neurons) organized in layers.",
                "Backpropagation is the key algorithm for training neural networks.",
                "Convolutional neural networks excel at image processing tasks."
            ],
            "data science": [
                "Data science combines statistics, programming, and domain expertise.",
                "The data science process includes collection, cleaning, analysis, and visualization.",
                "Python and R are the most popular programming languages for data science.",
                "Big data technologies enable processing of massive datasets."
            ]
        }
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[str]:
        """Simulate vector similarity search with delay"""
        self.query_count += 1
        
        # Simulate expensive query delay
        time.sleep(self.query_delay)
        
        # Simple keyword matching for demo
        query_lower = query.lower()
        relevant_docs = []
        
        for topic, docs in self.documents.items():
            if any(word in query_lower for word in topic.split()):
                relevant_docs.extend(docs)
        
        # If no specific match, return some general AI docs
        if not relevant_docs:
            relevant_docs = self.documents["artificial intelligence"]
        
        return relevant_docs[:top_k]
    
    def get_database_name(self) -> str:
        return self.name

def setup_cors_system():
    """Set up the complete CoRS system"""
    try:
        # Connect to Redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        redis_client.ping()
        print("✅ Connected to Redis")
    except:
        print("❌ Redis not available - using mock Redis")
        redis_client = MockRedis()
    
    # Import CoRS components
    from ..src.core.shared_dynamic_workspace import SharedDynamicWorkspace
    from ..src.core.cors_protocol import CoRSRetrievalProtocol
    from ..src.agents.cors_collaborative_agents import CollaborativeTeam
    
    # Initialize components
    workspace = SharedDynamicWorkspace(
        redis_client=redis_client,
        default_ttl=3600  # 1 hour
    )
    
    vector_store = MockVectorStore()
    
    cors_protocol = CoRSRetrievalProtocol(
        workspace=workspace,
        vector_store=vector_store
    )
    
    # Initialize LLM (use mock for demo if no API key)
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0.1)
    except:
        llm = MockLLM()
    
    # Create collaborative team
    team = CollaborativeTeam(cors_protocol, llm)
    
    return team, cors_protocol, workspace, vector_store

class MockRedis:
    """Mock Redis for demo when Redis is not available"""
    def __init__(self):
        self.data = {}
        self.counters = {}
    
    def hset(self, key, mapping=None, **kwargs):
        if key not in self.data:
            self.data[key] = {}
        if mapping:
            self.data[key].update(mapping)
        self.data[key].update(kwargs)
    
    def hgetall(self, key):
        return self.data.get(key, {})
    
    def keys(self, pattern):
        return [k for k in self.data.keys() if pattern.replace('*', '') in k]
    
    def delete(self, *keys):
        for key in keys:
            self.data.pop(key, None)
    
    def hincrby(self, key, field, amount=1):
        if key not in self.data:
            self.data[key] = {}
        current = int(self.data[key].get(field, 0))
        self.data[key][field] = str(current + amount)
    
    def expire(self, key, seconds):
        pass  # Mock implementation
    
    def set(self, key, value, ex=None):
        self.data[key] = value
    
    def get(self, key):
        return self.data.get(key)
    
    def ping(self):
        return True

class MockLLM:
    """Mock LLM for demo when OpenAI API is not available"""
    def invoke(self, messages, **kwargs):
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        # Generate a simple mock response based on the last message
        last_message = messages[-1].content if messages else "No content"
        
        if "research" in last_message.lower():
            return MockResponse("Based on the available information, this appears to be a comprehensive topic requiring further analysis. Key findings include multiple relevant sources that provide foundational knowledge. This research demonstrates the collaborative nature of information gathering.")
        elif "plan" in last_message.lower():
            return MockResponse("""
OBJECTIVE ANALYSIS:
The research objective requires systematic investigation across multiple domains.

SUBTASK BREAKDOWN:
1. Literature review (Priority 1)
2. Technical analysis (Priority 2) 
3. Current trends assessment (Priority 3)
4. Gap identification (Priority 4)

EXECUTION STRATEGY:
Parallel research execution with knowledge sharing through collaborative workspace.

RESOURCE ALLOCATION:
Distribute tasks among specialized researchers for maximum efficiency.

SUCCESS CRITERIA:
Comprehensive coverage with verified information from multiple sources.
""")
        elif "synthesis" in last_message.lower():
            return MockResponse("This synthesis integrates verified findings from multiple collaborative sources. The research demonstrates strong consensus across different perspectives, with key insights emerging from the collective intelligence of the research team. The collaborative verification process ensures high reliability of the conclusions presented.")
        else:
            return MockResponse("This is a mock response for demonstration purposes.")

def demonstrate_traditional_rag():
    """Demonstrate traditional multi-agent RAG where each agent works independently"""
    print("\n" + "="*60)
    print("🔄 TRADITIONAL MULTI-AGENT RAG DEMONSTRATION")
    print("="*60)
    
    vector_store = MockVectorStore("TraditionalKB")
    start_time = time.time()
    
    # Simulate 3 agents researching the same topic independently
    research_topics = [
        "What is artificial intelligence and machine learning?",
        "How do neural networks work in AI systems?", 
        "What are the applications of machine learning?"
    ]
    
    total_queries = 0
    results = []
    
    for i, topic in enumerate(research_topics):
        print(f"\n🤖 Agent {i+1} researching: {topic}")
        
        # Each agent queries independently (no sharing)
        chunks = vector_store.similarity_search(topic, top_k=3)
        total_queries += 1
        
        results.append({
            'agent_id': f'traditional_agent_{i+1}',
            'topic': topic,
            'chunks_found': len(chunks),
            'unique_info': True  # All info is "unique" since no sharing
        })
        
        print(f"   📚 Found {len(chunks)} chunks from knowledge base")
    
    total_time = time.time() - start_time
    
    print(f"\n📊 TRADITIONAL RAG RESULTS:")
    print(f"   ⏱️  Total time: {total_time:.2f}s")
    print(f"   🔍 Total KB queries: {total_queries}")
    print(f"   💰 Estimated cost: ${total_queries * 0.10:.2f}")
    print(f"   🔄 Information reuse: 0% (no sharing)")
    print(f"   🤝 Collaboration efficiency: 0% (independent work)")
    
    return {
        'approach': 'traditional',
        'total_time': total_time,
        'kb_queries': total_queries,
        'information_reuse': 0.0,
        'collaboration_efficiency': 0.0
    }

def demonstrate_cors_collaborative():
    """Demonstrate CoRS collaborative approach with shared workspace"""
    print("\n" + "="*60)
    print("🚀 CoRS COLLABORATIVE RAG DEMONSTRATION")
    print("="*60)
    
    try:
        team, cors_protocol, workspace, vector_store = setup_cors_system()
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return None
    
    # Clear workspace for clean demo
    workspace.clear_workspace()
    
    # Add team members
    team.add_researcher("ai_researcher", "artificial intelligence")
    team.add_researcher("ml_researcher", "machine learning") 
    team.add_researcher("data_researcher", "data science")
    team.add_planner("strategic_planner")
    team.add_synthesizer("knowledge_synthesizer")
    
    start_time = time.time()
    
    print("\n🎯 Phase 1: First researcher explores AI landscape")
    ai_researcher = team.agents["ai_researcher"]
    result1 = ai_researcher.research_topic("What is artificial intelligence and its core concepts?")
    
    print(f"   📊 Cache hit: {result1['cache_hit']} (expected: False - first query)")
    print(f"   📚 KB queries so far: {vector_store.query_count}")
    
    print("\n🎯 Phase 2: Second researcher explores ML (should find shared info)")
    ml_researcher = team.agents["ml_researcher"]  
    result2 = ml_researcher.research_topic("How does machine learning relate to artificial intelligence?")
    
    print(f"   📊 Cache hit: {result2['cache_hit']} (expected: True - shared workspace)")
    print(f"   📚 KB queries so far: {vector_store.query_count}")
    
    print("\n🎯 Phase 3: Third researcher explores data science (more sharing expected)")
    data_researcher = team.agents["data_researcher"]
    result3 = data_researcher.research_topic("What role does data science play in AI and ML?")
    
    print(f"   📊 Cache hit: {result3['cache_hit']}")
    print(f"   📚 KB queries so far: {vector_store.query_count}")
    
    print("\n🎯 Phase 4: Planner creates research strategy (using team knowledge)")
    planner = team.agents["strategic_planner"]
    plan_result = planner.create_research_plan("Comprehensive AI/ML research initiative")
    
    print(f"   🧠 Leveraged existing knowledge: {plan_result['existing_knowledge_leveraged']}")
    print(f"   📚 KB queries so far: {vector_store.query_count}")
    
    print("\n🎯 Phase 5: Synthesizer creates final report (from verified workspace)")
    synthesizer = team.agents["knowledge_synthesizer"]
    synthesis_result = synthesizer.synthesize_research("Artificial Intelligence and Machine Learning Overview")
    
    print(f"   🔗 Chunks synthesized: {synthesis_result['chunks_synthesized']}")
    print(f"   ⭐ Avg verification: {synthesis_result['verification_quality']:.1f}")
    print(f"   👥 Team contributors: {synthesis_result['team_contributors']}")
    
    total_time = time.time() - start_time
    
    # Get comprehensive metrics
    workspace_stats = workspace.get_workspace_stats()
    protocol_metrics = cors_protocol.get_protocol_metrics()
    
    print(f"\n📊 CoRS COLLABORATIVE RESULTS:")
    print(f"   ⏱️  Total time: {total_time:.2f}s")
    print(f"   🔍 Total KB queries: {vector_store.query_count}")
    print(f"   💰 Estimated cost: ${vector_store.query_count * 0.10:.2f}")
    print(f"   🎯 Cache hit rate: {protocol_metrics.cache_hit_rate * 100:.1f}%")
    print(f"   🔄 Information reuse: {workspace_stats.avg_verification_count:.1f}x")
    print(f"   🤝 Collaboration score: {workspace_stats.collaboration_score:.2f}")
    print(f"   📈 Synthesis efficiency: {workspace_stats.synthesis_efficiency * 100:.1f}%")
    
    return {
        'approach': 'cors_collaborative',
        'total_time': total_time,
        'kb_queries': vector_store.query_count,
        'cache_hit_rate': protocol_metrics.cache_hit_rate,
        'information_reuse': workspace_stats.avg_verification_count,
        'collaboration_efficiency': workspace_stats.collaboration_score,
        'synthesis_efficiency': workspace_stats.synthesis_efficiency
    }

def compare_approaches(traditional_results: Dict, cors_results: Dict):
    """Compare traditional vs CoRS approaches"""
    print("\n" + "="*60)
    print("📈 PERFORMANCE COMPARISON")
    print("="*60)
    
    if not cors_results:
        print("❌ CoRS demonstration failed - cannot compare")
        return
    
    # Calculate improvements
    query_reduction = (traditional_results['kb_queries'] - cors_results['kb_queries']) / traditional_results['kb_queries']
    time_improvement = (traditional_results['total_time'] - cors_results['total_time']) / traditional_results['total_time']
    cost_savings = query_reduction * 100
    
    print(f"\n🎯 EFFICIENCY GAINS:")
    print(f"   📉 Query reduction: {query_reduction * 100:.1f}%")
    print(f"   ⚡ Time improvement: {time_improvement * 100:.1f}%") 
    print(f"   💰 Cost savings: {cost_savings:.1f}%")
    
    print(f"\n🤝 COLLABORATION BENEFITS:")
    print(f"   🔄 Information reuse: {cors_results['information_reuse']:.1f}x verification")
    print(f"   🎯 Cache hit rate: {cors_results['cache_hit_rate'] * 100:.1f}%")
    print(f"   🏆 Collaboration score: {cors_results['collaboration_efficiency']:.2f}")
    
    print(f"\n📊 QUALITY IMPROVEMENTS:")
    print(f"   ✅ Synthesis efficiency: {cors_results['synthesis_efficiency'] * 100:.1f}%")
    print(f"   🔍 Emergent consensus: Automatic through verification counts")
    print(f"   📋 Audit trail: Complete provenance tracking")
    
    print(f"\n🎉 KEY INSIGHTS:")
    print(f"   • CoRS transforms RAG from solitary to collaborative")
    print(f"   • Workspace acts as intelligent cache with metadata")
    print(f"   • Agents build upon each other's discoveries")
    print(f"   • Verification counts create automatic consensus")
    print(f"   • Massive efficiency gains through knowledge sharing")

def demonstrate_workspace_intelligence():
    """Show the intelligence and metadata in the workspace"""
    print("\n" + "="*60)
    print("🧠 WORKSPACE INTELLIGENCE DEMONSTRATION")
    print("="*60)
    
    try:
        _, cors_protocol, workspace, _ = setup_cors_system()
        
        # Get workspace intelligence report
        intelligence_report = cors_protocol.get_workspace_intelligence_report()
        
        print("\n📊 Workspace Statistics:")
        ws_stats = intelligence_report.get('workspace_intelligence', {})
        print(f"   📚 Total chunks: {ws_stats.get('total_chunks', 0)}")
        print(f"   ⭐ Avg verification: {ws_stats.get('avg_verification_count', 0):.1f}")
        print(f"   🤝 Collaboration score: {ws_stats.get('collaboration_score', 0):.2f}")
        print(f"   📈 Synthesis efficiency: {ws_stats.get('synthesis_efficiency', 0) * 100:.1f}%")
        
        print("\n🎯 Protocol Performance:")
        protocol_perf = intelligence_report.get('protocol_performance', {})
        print(f"   🔍 Total queries: {protocol_perf.get('total_queries', 0)}")
        print(f"   🎯 Cache hit rate: {protocol_perf.get('cache_hit_rate', 0) * 100:.1f}%")
        print(f"   ⚡ Avg retrieval time: {protocol_perf.get('avg_retrieval_time', 0):.3f}s")
        
        print("\n💰 Efficiency Gains:")
        efficiency = intelligence_report.get('efficiency_gains', {})
        print(f"   🔄 Queries saved: {efficiency.get('queries_saved', 0)}")
        print(f"   📉 KB query reduction: {efficiency.get('kb_query_reduction', '0%')}")
        print(f"   💵 Estimated savings: ${efficiency.get('estimated_cost_savings', 0):.2f}")
        
    except Exception as e:
        print(f"❌ Workspace intelligence demo failed: {e}")

def main():
    """Run the complete CoRS demonstration"""
    print("🚀 CoRS Collaborative Retrieval and Synthesis Demonstration")
    print("This demo shows how CoRS transforms RAG into a collaborative team sport!")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Demonstrate traditional approach
        traditional_results = demonstrate_traditional_rag()
        
        # Demonstrate CoRS collaborative approach  
        cors_results = demonstrate_cors_collaborative()
        
        # Compare the two approaches
        compare_approaches(traditional_results, cors_results)
        
        # Show workspace intelligence
        demonstrate_workspace_intelligence()
        
        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("="*60)
        print("\nKey Takeaways:")
        print("• CoRS enables true collaborative intelligence")
        print("• Workspace acts as shared memory with rich metadata") 
        print("• Agents build upon each other's discoveries")
        print("• Massive efficiency gains through intelligent caching")
        print("• Emergent consensus through verification counts")
        print("• Complete auditability and transparency")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()