"""
Streamlit demo for the Agentic Deep Research System.

Run with: streamlit run app/streamlit_demo.py
"""
import streamlit as st
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import config
from src.agent import ResearchAgent, AgentResult
from src.evaluation import ABLATION_CONFIGS

# Page config
st.set_page_config(
    page_title="Agentic Deep Research",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stTextArea textarea {
    font-size: 16px;
}
.citation-box {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.trace-step {
    border-left: 3px solid #4CAF50;
    padding-left: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🔬 Agentic Deep Research System")
    st.markdown("*Research assistant powered by LLM agents over arXiv papers*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        config_name = st.selectbox(
            "Agent Configuration",
            options=list(ABLATION_CONFIGS.keys()),
            index=0,
            help="Select the agent configuration to use"
        )
        
        st.markdown("---")
        st.markdown("### Configuration Details")
        config_details = ABLATION_CONFIGS[config_name]
        for key, value in config_details.items():
            st.text(f"{key}: {value}")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This demo showcases an agentic deep research system built for 
        the AIMS-DTU Research Intern 2026 assignment.
        
        **Features:**
        - Query decomposition & planning
        - Hybrid retrieval (semantic + lexical)
        - Iterative reflection loop
        - Citation verification
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Research Query")
        
        # Example queries
        example_queries = [
            "What is the ReAct framework and how does it combine reasoning with acting?",
            "Compare ReAct and Reflexion - how do their approaches differ?",
            "What are the main architectural patterns used in LLM agent systems?",
            "How do researchers address hallucination issues in LLM agents?"
        ]
        
        selected_example = st.selectbox(
            "Example queries:",
            options=[""] + example_queries,
            index=0
        )
        
        query = st.text_area(
            "Enter your research question:",
            value=selected_example if selected_example else "",
            height=100,
            placeholder="e.g., What are the main approaches to tool use in LLM agents?"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_button = st.button("🚀 Run Research", type="primary", use_container_width=True)
        with col_btn2:
            show_trace = st.checkbox("Show execution trace", value=True)
    
    with col2:
        st.header("📊 Quick Stats")
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            st.metric("Confidence", f"{result.confidence:.2f}")
            st.metric("Iterations", result.iterations)
            st.metric("Tool Calls", result.tool_calls)
            st.metric("Latency", f"{result.latency_seconds:.1f}s")
    
    # Run research
    if run_button and query:
        with st.spinner("Researching... This may take a moment."):
            try:
                # Create agent with selected config
                agent = ResearchAgent.create_from_ablation(config_name)
                result = agent.research(query)
                
                # Store result in session state
                st.session_state['last_result'] = result
                st.session_state['last_query'] = query
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return
    
    # Display results
    if 'last_result' in st.session_state:
        result = st.session_state['last_result']
        
        st.markdown("---")
        st.header("📄 Answer")
        
        # Format answer with clickable citations
        answer = result.answer
        for arxiv_id in result.citations:
            answer = answer.replace(
                f"[arXiv:{arxiv_id}]",
                f"[[arXiv:{arxiv_id}]](https://arxiv.org/abs/{arxiv_id})"
            )
        
        st.markdown(answer)
        
        # Citations section
        st.markdown("---")
        st.header("📚 Citations")
        
        if result.citations:
            for arxiv_id in result.citations:
                st.markdown(f"""
                <div class="citation-box">
                    <strong>arXiv:{arxiv_id}</strong><br>
                    <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank">View on arXiv</a> | 
                    <a href="https://arxiv.org/pdf/{arxiv_id}.pdf" target="_blank">PDF</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No citations in this answer.")
        
        # Execution trace
        if show_trace and result.trace:
            st.markdown("---")
            st.header("🔍 Execution Trace")
            
            trace = result.trace
            if isinstance(trace, dict) and 'steps' in trace:
                for i, step in enumerate(trace['steps'], 1):
                    with st.expander(f"Step {i}: {step['type'].upper()}", expanded=False):
                        st.json(step)
        
        # Download results
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            result_json = json.dumps(result.to_dict(), indent=2)
            st.download_button(
                "📥 Download Full Result (JSON)",
                result_json,
                file_name="research_result.json",
                mime="application/json"
            )
        
        with col_dl2:
            submission = result.to_submission_format()
            submission_json = json.dumps(submission, indent=2)
            st.download_button(
                "📥 Download Submission Format",
                submission_json,
                file_name="submission.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
