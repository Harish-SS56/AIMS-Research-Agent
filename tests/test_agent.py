"""
Unit tests for the Agentic Deep Research System.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPlanner:
    """Tests for the planner module."""
    
    def test_plan_factoid_question(self):
        """Test planning for a factoid question."""
        from src.agent.planner import Planner
        
        planner = Planner()
        plan = planner.plan("What is the ReAct framework?")
        
        assert plan.original_query == "What is the ReAct framework?"
        assert plan.query_type in ["factoid", "comparative", "survey"]
        assert len(plan.sub_questions) >= 1
        assert len(plan.search_queries) >= 1
    
    def test_plan_comparative_question(self):
        """Test planning for a comparative question."""
        from src.agent.planner import Planner
        
        planner = Planner()
        plan = planner.plan("Compare ReAct and Reflexion approaches")
        
        assert plan.query_type in ["factoid", "comparative", "survey"]
        assert len(plan.sub_questions) >= 1


class TestRetrieval:
    """Tests for retrieval components."""
    
    def test_embedding_generation(self):
        """Test embedding generation."""
        from src.retrieval.embeddings import embed_single
        
        embedding = embed_single("test query")
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    def test_citation_extraction(self):
        """Test citation extraction from text."""
        from src.agent.synthesizer import extract_citations
        
        text = "This is shown in [arXiv:2210.03629] and [arXiv:2303.11366]."
        citations = extract_citations(text)
        
        assert "2210.03629" in citations
        assert "2303.11366" in citations


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_citation_precision(self):
        """Test citation precision calculation."""
        from src.evaluation.metrics import calculate_citation_precision
        
        # All predicted are relevant
        precision = calculate_citation_precision(
            ["a", "b"],
            ["a", "b", "c"]
        )
        assert precision == 1.0
        
        # Half are relevant
        precision = calculate_citation_precision(
            ["a", "b"],
            ["a"]
        )
        assert precision == 0.5
        
        # Empty prediction
        precision = calculate_citation_precision([], ["a", "b"])
        assert precision == 1.0
    
    def test_citation_recall(self):
        """Test citation recall calculation."""
        from src.evaluation.metrics import calculate_citation_recall
        
        # All must-cite are cited
        recall = calculate_citation_recall(
            ["a", "b", "c"],
            ["a", "b"]
        )
        assert recall == 1.0
        
        # Half of must-cite are cited
        recall = calculate_citation_recall(
            ["a"],
            ["a", "b"]
        )
        assert recall == 0.5
        
        # No must-cite
        recall = calculate_citation_recall(["a", "b"], [])
        assert recall == 1.0


class TestAgent:
    """Tests for the research agent."""
    
    def test_agent_creation(self):
        """Test agent instantiation."""
        from src.agent import ResearchAgent
        
        agent = ResearchAgent()
        assert agent.use_planner == True
        assert agent.use_reflector == True
    
    def test_baseline_creation(self):
        """Test baseline agent creation."""
        from src.agent import ResearchAgent
        
        agent = ResearchAgent.create_baseline()
        assert agent.use_planner == False
        assert agent.use_reflector == False
        assert agent.max_iterations == 1
    
    def test_ablation_creation(self):
        """Test ablation agent creation."""
        from src.agent import ResearchAgent
        
        agent = ResearchAgent.create_from_ablation("no_planner")
        assert agent.use_planner == False
        assert agent.use_reflector == True
        
        agent = ResearchAgent.create_from_ablation("no_hybrid")
        assert agent.use_hybrid == False


class TestConfig:
    """Tests for configuration."""
    
    def test_config_loading(self):
        """Test configuration loads correctly."""
        from src.utils.config import config
        
        assert config.azure_openai.chat_deployment == "gpt-4o"
        assert config.chunking.chunk_size == 512
    
    def test_ablation_config(self):
        """Test ablation configuration."""
        from src.utils.config import Config
        
        config = Config.from_ablation("baseline")
        assert config.agent.use_planner == False
        assert config.agent.use_reflector == False
        assert config.agent.max_iterations == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
