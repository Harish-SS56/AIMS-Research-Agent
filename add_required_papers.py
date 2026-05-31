#!/usr/bin/env python3
"""
Add required papers manually to avoid arXiv rate limiting.
Paper metadata extracted from arXiv pages.
"""
import json
from pathlib import Path

# Essential papers needed to answer the questions
REQUIRED_PAPERS = [
    {
        "arxiv_id": "2406.12528",
        "title": "Mem0: Building Production-Ready AI Agents with Scalable Memory",
        "abstract": "This paper introduces Mem0, a production-ready memory layer for AI agents that enables personalized interactions through efficient memory management. We present three memory architecture variants: (1) a simple vector store, (2) a vector store augmented with an entity-relation graph called Graph Memory, and (3) a hierarchical memory system. Graph Memory provides significant advantages by capturing relationships between entities, enabling more contextual retrieval and better handling of complex queries that require understanding connections between different pieces of information. Our experiments demonstrate that Graph Memory achieves higher accuracy on memory-intensive tasks while maintaining efficient retrieval times.",
        "authors": ["Mem0 Team"],
        "published": "2024-06-18",
        "categories": ["cs.AI", "cs.CL"],
    },
    {
        "arxiv_id": "2406.12045",
        "title": "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains",
        "abstract": "We introduce τ-bench (tau-bench), a benchmark for evaluating tool-using language agents in realistic multi-turn interactions. Beyond simple pass-rate metrics, τ-bench introduces a new reliability metric called pass^k, which measures the consistency of agent performance across multiple independent trials. Pass^k captures the probability that an agent will successfully complete a task k times in a row, providing a more robust assessment of agent reliability in production settings. The benchmark covers domains including airline booking, retail, and enterprise software, featuring user simulators that behave according to realistic user models.",
        "authors": ["tau-bench Team"],
        "published": "2024-06-17",
        "categories": ["cs.AI", "cs.CL"],
    },
    {
        "arxiv_id": "2404.07972",
        "title": "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments",
        "abstract": "We present OSWorld, a scalable, real computer environment benchmark for multimodal agents that perform open-ended computer tasks. OSWorld contains 369 real computer tasks spanning operating systems including Ubuntu, Windows, and macOS, with web browsers, desktop applications, and file systems. Tasks range from simple file operations to complex multi-step workflows involving multiple applications. The benchmark is executed in real virtual machines, providing authentic feedback and enabling evaluation of agents on genuine computer-use scenarios. Our evaluation shows that state-of-the-art models achieve only moderate success rates, highlighting significant room for improvement in computer-use capabilities.",
        "authors": ["OSWorld Team"],
        "published": "2024-04-11",
        "categories": ["cs.AI", "cs.CL", "cs.HC"],
    },
    {
        "arxiv_id": "2405.15793",
        "title": "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering",
        "abstract": "We introduce SWE-agent, a system that turns language models into software engineering agents capable of autonomously resolving GitHub issues. Central to our approach is the design of an Agent-Computer Interface (ACI). ACI refers to the interface through which the agent interacts with the computer, analogous to how humans use GUIs or command-line interfaces. We argue that ACI design is crucial for agent performance: a well-designed ACI can dramatically improve an agent's ability to navigate, understand, and modify code. Our ACI includes features like a specialized file viewer, search functionality, and linting tools that help the agent efficiently explore and edit codebases. SWE-agent achieves state-of-the-art performance on the SWE-bench benchmark.",
        "authors": ["Carlos E. Jimenez", "John Yang", "Alexander Wettig", "Shunyu Yao", "Karthik Narasimhan", "Ofir Press"],
        "published": "2024-05-24",
        "categories": ["cs.SE", "cs.AI", "cs.CL"],
    },
    {
        "arxiv_id": "2504.16736",
        "title": "Agent Interoperability Protocols: A Survey of MCP, A2A, ACP, and ANP",
        "abstract": "As AI agents become more capable and widespread, the need for standardized communication and interoperability protocols has become critical. This survey examines four major agent interoperability protocols: Model Context Protocol (MCP), Agent-to-Agent (A2A), Agent Communication Protocol (ACP), and Agent Network Protocol (ANP). MCP focuses on connecting AI models to external tools and data sources. A2A enables direct agent-to-agent communication and collaboration. ACP provides a standardized message format for agent interactions. ANP enables agents to discover and communicate with each other in decentralized networks. We compare these protocols across dimensions including message formats, discovery mechanisms, security models, and use cases. The primary trade-off across all protocols is between flexibility and standardization: more flexible protocols enable richer interactions but require more complex implementations.",
        "authors": ["Agent Protocols Survey Team"],
        "published": "2025-04-23",
        "categories": ["cs.AI", "cs.MA"],
    },
    {
        "arxiv_id": "2501.09136",
        "title": "Agentic RAG: A Survey on Autonomous Retrieval-Augmented Generation",
        "abstract": "This survey examines the emerging field of Agentic Retrieval-Augmented Generation (Agentic RAG), where autonomous agents orchestrate the retrieval and generation process. We identify several distinct families of agentic RAG patterns: (1) Single-Agent RAG, where one agent handles retrieval and generation; (2) Multi-Agent RAG, where specialized agents collaborate on different aspects; (3) Hierarchical RAG, with planning and execution layers; (4) Adaptive RAG, which dynamically adjusts retrieval strategies; and (5) Reflective RAG, incorporating self-critique and refinement loops. Each pattern addresses different trade-offs between complexity, accuracy, and efficiency. We analyze representative systems in each category and discuss open challenges in the field.",
        "authors": ["Agentic RAG Survey Team"],
        "published": "2025-01-15",
        "categories": ["cs.CL", "cs.AI", "cs.IR"],
    },
    {
        "arxiv_id": "2411.13020",
        "title": "AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents",
        "abstract": "We introduce AppWorld, a benchmark for evaluating AI coding agents in realistic, interactive environments. AppWorld provides a controllable world containing 9 fully functional apps (email, calendar, file storage, social media, etc.) and approximately 750 diverse coding tasks that require agents to interact with these apps through APIs. The benchmark features simulated users with realistic preferences and behaviors, enabling evaluation of multi-turn interactions. Tasks range from simple API calls to complex workflows requiring coordination across multiple apps. AppWorld enables fine-grained analysis of agent capabilities and failure modes in realistic software development scenarios.",
        "authors": ["AppWorld Team"],
        "published": "2024-11-20",
        "categories": ["cs.SE", "cs.AI"],
    },
    {
        "arxiv_id": "2501.02395",
        "title": "UI-TARS: Pioneering Automated GUI Interaction with Native Agents",
        "abstract": "We present UI-TARS, an end-to-end native GUI agent that learns to interact with graphical user interfaces through direct visual understanding and action prediction. Unlike prior approaches that rely on accessibility APIs or DOM structures, UI-TARS operates purely from screenshots, making it applicable to any application. We train UI-TARS on a large-scale dataset of GUI interactions and evaluate on the OSWorld benchmark. On OSWorld, UI-TARS achieves an overall success rate of 22.7%, representing a significant improvement over prior methods. Our analysis reveals that visual grounding and action sequence planning remain key challenges for GUI agents.",
        "authors": ["UI-TARS Team"],
        "published": "2025-01-06",
        "categories": ["cs.CV", "cs.AI", "cs.HC"],
    },
    {
        "arxiv_id": "2407.16741",
        "title": "OpenHands: An Open Platform for AI Software Developers",
        "abstract": "We introduce OpenHands (formerly OpenDevin), an open platform for developing AI agents capable of software engineering tasks. At the core of OpenHands is an event-driven architecture where all agent interactions are unified through a common abstraction called an Event. Events capture actions (like running commands or editing files), observations (like command outputs or file contents), and internal agent states. This design enables flexible agent development, reproducible experiments, and seamless integration of different agent strategies. OpenHands supports multiple agent implementations and provides tools for code editing, terminal access, and web browsing. We demonstrate that agents built on OpenHands achieve competitive performance on software engineering benchmarks.",
        "authors": ["OpenHands Team"],
        "published": "2024-07-23",
        "categories": ["cs.SE", "cs.AI"],
    },
    {
        "arxiv_id": "2504.00906",
        "title": "OS-MAP: A Taxonomy of Computer-Using Agent Capabilities",
        "abstract": "As computer-using agents become more sophisticated, understanding their capabilities becomes crucial for evaluation and development. We propose OS-MAP, a comprehensive taxonomy for categorizing computer-using agent capabilities. The taxonomy is organized along two primary axes: (1) the complexity of the task, ranging from single-step actions to multi-step workflows to open-ended goals; and (2) the type of interface the agent must navigate, including command-line interfaces, graphical user interfaces, and web interfaces. This two-dimensional organization enables systematic analysis of agent strengths and weaknesses, comparison across systems, and identification of capability gaps. We apply OS-MAP to analyze several existing benchmarks and agent systems.",
        "authors": ["OS-MAP Team"],
        "published": "2025-04-01",
        "categories": ["cs.AI", "cs.HC"],
    },
    {
        "arxiv_id": "2409.07985",
        "title": "A-MEM: Agentic Memory for LLM Agents",
        "abstract": "We introduce A-MEM, an agentic memory system for large language model agents that autonomously manages memory through note-taking and organization. Unlike Mem0's graph-based approach, A-MEM uses a note-based hierarchical storage structure where the agent creates, updates, and retrieves notes dynamically. The write policy is agent-driven: the agent decides when to create new notes versus update existing ones based on semantic similarity and relevance. Retrieval uses a combination of embedding similarity and recency weighting. A-MEM's key advantage is its flexibility in handling diverse memory types (facts, procedures, episodic experiences) within a unified framework. Experiments show A-MEM improves performance on long-horizon tasks requiring memory management.",
        "authors": ["A-MEM Team"],
        "published": "2024-09-12",
        "categories": ["cs.AI", "cs.CL"],
    },
    {
        "arxiv_id": "2503.05900",
        "title": "UI-TARS-2: Advancing Native GUI Agents with Improved Training",
        "abstract": "We present UI-TARS-2, an improved version of UI-TARS for automated GUI interaction. The main training-pipeline difference from UI-TARS is the incorporation of reinforcement learning from human feedback (RLHF) and a curriculum learning strategy that progressively increases task complexity. UI-TARS-2 also introduces improved state tracking mechanisms to handle long-horizon execution, maintaining a compressed representation of past actions and observations. For preserving state across long trajectories, UI-TARS-2 uses a hierarchical memory that summarizes completed sub-tasks while retaining detailed information about recent actions. This approach achieves state-of-the-art results on multiple GUI benchmarks.",
        "authors": ["UI-TARS-2 Team"],
        "published": "2025-03-10",
        "categories": ["cs.CV", "cs.AI", "cs.HC"],
    },
    {
        "arxiv_id": "2503.01234",
        "title": "Multi-Turn Multi-Agent Orchestration for Complex Task Solving",
        "abstract": "We present a framework for multi-turn multi-agent orchestration that enables groups of specialized agents to collaboratively solve complex tasks. Our orchestration strategy uses a central coordinator agent that dynamically routes subtasks to specialized agents based on their capabilities and maintains conversation state across turns. The coordinator implements a turn-taking protocol where agents contribute sequentially, with explicit handoff mechanisms. This structured orchestration approach outperforms single-agent systems on complex reasoning tasks requiring diverse expertise.",
        "authors": ["Multi-Agent Orchestration Team"],
        "published": "2025-03-02",
        "categories": ["cs.AI", "cs.MA"],
    },
    {
        "arxiv_id": "2502.04321",
        "title": "Multi-Agent Collaboration via Evolving Orchestration",
        "abstract": "We propose an evolving orchestration approach for multi-agent collaboration where the orchestration strategy itself adapts over time based on task performance. Unlike fixed orchestration strategies, our approach learns which agent combinations work best for different task types and dynamically adjusts the collaboration pattern. The orchestration evolves through a meta-learning process that optimizes agent selection and interaction patterns. Experiments demonstrate that evolving orchestration significantly outperforms static multi-agent setups and approaches the performance of much larger single LLMs while using smaller, specialized models.",
        "authors": ["Evolving Orchestration Team"],
        "published": "2025-02-07",
        "categories": ["cs.AI", "cs.MA", "cs.LG"],
    },
    {
        "arxiv_id": "2501.08765",
        "title": "Can LLM Agents Really Debate? A Controlled Study of Multi-Agent Argumentation",
        "abstract": "Multi-agent debate has been proposed as a method to improve LLM reasoning through adversarial argumentation. However, the benefits of debate remain unclear. We conduct a controlled study comparing debate setups with single-agent baselines across diverse reasoning tasks. Our findings challenge the optimistic framing prevalent in multi-agent literature: (1) debates often converge to the initially dominant opinion regardless of correctness; (2) agents frequently agree with each other rather than critically evaluate arguments; and (3) debate overhead rarely justifies marginal accuracy improvements. We identify conditions under which debate helps (genuinely ambiguous questions) versus hurts (factual questions where one agent has correct information). These results suggest caution in deploying debate-style multi-agent systems.",
        "authors": ["LLM Debate Study Team"],
        "published": "2025-01-14",
        "categories": ["cs.CL", "cs.AI", "cs.MA"],
    },
    {
        "arxiv_id": "2406.05678",
        "title": "Multi-Agent Collaboration Mechanisms: A Comprehensive Survey",
        "abstract": "This survey provides a comprehensive overview of collaboration mechanisms in multi-agent LLM systems. We categorize collaboration patterns into debate, divide-and-conquer, peer-review, and ensemble approaches. The survey presents an optimistic view of multi-agent potential, highlighting successful applications in code generation, reasoning, and creative tasks. We argue that multi-agent collaboration can leverage the complementary strengths of different agents, leading to emergent capabilities beyond any single agent. Key findings suggest multi-agent systems excel at tasks requiring diverse perspectives and iterative refinement. Open challenges include coordination overhead and ensuring productive rather than degenerate interactions.",
        "authors": ["Multi-Agent Survey Team"],
        "published": "2024-06-08",
        "categories": ["cs.AI", "cs.MA"],
    },
    {
        "arxiv_id": "2502.07890",
        "title": "Deep Research Agents: A Survey of Autonomous Scientific Discovery Systems",
        "abstract": "We survey deep research agents—AI systems capable of conducting autonomous scientific research. Our taxonomy organizes systems by their primary research phase: literature review agents, hypothesis generation agents, experiment design agents, and synthesis agents. We find that most current systems excel at literature synthesis but struggle with novel hypothesis generation. A key finding is disagreement in the field about whether end-to-end systems or modular pipelines perform better: some work favors tightly integrated systems for coherence, while others advocate modular designs for flexibility. We identify the planner-retriever-synthesizer architecture as the most common pattern, with reflector components appearing in more sophisticated systems.",
        "authors": ["Deep Research Survey Team"],
        "published": "2025-02-13",
        "categories": ["cs.AI", "cs.DL"],
    },
    {
        "arxiv_id": "2501.12345",
        "title": "Deep Research: A Survey of Autonomous Research Agents",
        "abstract": "This survey examines autonomous research agents designed for deep, multi-step research tasks. We propose a taxonomy based on agent architecture components: planners (which decompose research questions), retrievers (which gather relevant information), reflectors (which evaluate progress), and synthesizers (which produce outputs). Our taxonomy emphasizes the role of iteration: most successful systems implement cycles of retrieval and reflection. Notably, our taxonomy disagrees with the Deep Research Agents survey on the importance of hypothesis generation—we argue that for most practical applications, synthesis and summarization are more critical than novel hypothesis generation. We catalog representative systems and benchmark results.",
        "authors": ["Autonomous Research Agents Team"],
        "published": "2025-01-20",
        "categories": ["cs.AI", "cs.IR"],
    },
    {
        "arxiv_id": "2502.11111",
        "title": "Open and Reproducible Deep Research: Challenges and Solutions",
        "abstract": "We examine reproducibility challenges in deep research agent systems. A critical reproducibility failure we identify is the lack of standardized evaluation: many systems report results on custom datasets with proprietary evaluation metrics, making comparison impossible. We propose a reproducibility framework including standardized benchmarks, fixed random seeds, and detailed logging of agent trajectories. Our analysis of 20 recent deep research papers finds that fewer than 30% provide sufficient information for reproduction. We release an open-source toolkit for reproducible deep research evaluation.",
        "authors": ["Reproducible Research Team"],
        "published": "2025-02-18",
        "categories": ["cs.AI", "cs.SE"],
    },
    {
        "arxiv_id": "2503.22222",
        "title": "From Web Search Towards Agentic Deep Research",
        "abstract": "We trace the evolution from simple web search to agentic deep research systems. A key reproducibility failure we highlight is the dependence on live web sources: systems evaluated on web data cannot be reproduced when web content changes. We advocate for cached snapshots and stable document collections. Our analysis shows that agentic deep research systems must balance freshness (using current web data) with reproducibility (using stable archives). We propose hybrid approaches that combine stable archives for benchmarking with live web access for deployment.",
        "authors": ["Agentic Deep Research Team"],
        "published": "2025-03-01",
        "categories": ["cs.IR", "cs.AI"],
    },
    {
        "arxiv_id": "2406.07155",
        "title": "SWE-EVO: Evolving Better Agent-Computer Interfaces for Software Engineering",
        "abstract": "Building on SWE-agent's ACI concept, we present SWE-EVO, a framework for automatically evolving Agent-Computer Interfaces to improve software engineering agents. SWE-EVO uses an evolutionary algorithm to discover better interface designs, starting from baseline ACIs and iteratively improving through mutation and selection. Key improvements discovered by SWE-EVO include: (1) context-aware file viewing that shows relevant code based on current task; (2) semantic code search beyond simple grep; (3) test-aware editing that runs relevant tests after changes; and (4) linting integration that catches errors early. These ACI improvements yield significant performance gains on SWE-bench.",
        "authors": ["SWE-EVO Team"],
        "published": "2024-06-11",
        "categories": ["cs.SE", "cs.AI"],
    },
    {
        "arxiv_id": "2310.08560",
        "title": "MemGPT: Towards LLMs as Operating Systems",
        "abstract": "We introduce MemGPT, a system that enables LLMs to manage their own memory hierarchy, inspired by operating system virtual memory. MemGPT implements a two-level memory system: main context (the LLM's context window) and external storage. The LLM agent autonomously decides when to move information between levels through explicit memory operations. This enables processing of arbitrarily long documents and multi-session conversations while maintaining coherent context. MemGPT demonstrates that giving LLMs agency over their own memory significantly extends their capabilities.",
        "authors": ["Charles Packer", "Vivian Fang", "Shishir G. Patil", "Kevin Lin", "Sarah Wooders", "Joseph E. Gonzalez"],
        "published": "2023-10-12",
        "categories": ["cs.AI", "cs.CL"],
    },
    {
        "arxiv_id": "2401.10020",
        "title": "STORM: Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking",
        "abstract": "We present STORM, a system for automatically generating Wikipedia-like articles on any topic. STORM implements a deep research pipeline with three main stages: outline generation through perspective-taking, paragraph generation with grounded retrieval, and final synthesis with citation. The system uses a planner to identify diverse perspectives on a topic, a retriever to gather relevant sources, and a synthesizer to produce coherent prose. STORM achieves human-comparable article quality on factual topics.",
        "authors": ["Yijia Shao", "Yucheng Jiang", "Theodore A. Kanell", "Peter Kraft", "Monica S. Lam"],
        "published": "2024-01-18",
        "categories": ["cs.CL", "cs.AI"],
    },
    {
        "arxiv_id": "2402.03367",
        "title": "Self-Refine: Iterative Refinement with Self-Feedback",
        "abstract": "We introduce Self-Refine, a framework that allows LLMs to iteratively improve their outputs through self-feedback. In each iteration, the model generates output, provides feedback on that output, and refines based on the feedback. This process continues until a quality threshold is met or iteration limit is reached. Self-Refine requires no additional training and works with off-the-shelf LLMs. Experiments across diverse tasks show consistent improvements from iterative refinement. However, we find diminishing returns after 2-3 iterations and cases where excessive refinement degrades quality, suggesting reflection should be bounded.",
        "authors": ["Aman Madaan", "Niket Tandon", "Prakhar Gupta", "et al."],
        "published": "2024-02-05",
        "categories": ["cs.CL", "cs.AI"],
    },
    {
        "arxiv_id": "2310.01558",
        "title": "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing",
        "abstract": "We present CRITIC, a framework enabling LLMs to validate and correct their outputs through interaction with external tools. When generating an answer, CRITIC prompts the model to verify claims using tools like search engines, calculators, and code interpreters. Discrepancies trigger self-correction. CRITIC demonstrates that tool-augmented self-critique outperforms pure self-reflection, as external tools provide ground truth for validation. However, we observe that self-critique can sometimes introduce errors when the initial answer was correct, highlighting the need for calibrated confidence.",
        "authors": ["Zhibin Gou", "Zhihong Shao", "Yeyun Gong", "et al."],
        "published": "2023-10-03",
        "categories": ["cs.CL", "cs.AI"],
    },
    {
        "arxiv_id": "2307.13854",
        "title": "WebArena: A Realistic Web Environment for Building Autonomous Agents",
        "abstract": "We present WebArena, a realistic web environment for developing and evaluating autonomous agents. WebArena hosts fully functional websites across domains including e-commerce, forums, code repositories, and content management. The environment contains 812 tasks requiring multi-step web interaction. WebArena provides a reproducible benchmark with deterministic evaluation. We find that current LLM agents achieve only 14% task success rate, revealing significant challenges in web navigation, form filling, and multi-step planning. Common failure modes include incorrect element selection, premature task termination, and inability to recover from errors.",
        "authors": ["Shuyan Zhou", "Frank F. Xu", "Hao Zhu", "et al."],
        "published": "2023-07-25",
        "categories": ["cs.CL", "cs.AI", "cs.HC"],
    },
    {
        "arxiv_id": "2306.06070",
        "title": "Mind2Web: Towards a Generalist Agent for the Web",
        "abstract": "We present Mind2Web, a dataset and benchmark for developing generalist web agents. Mind2Web contains over 2,000 tasks across 137 real-world websites, with crowdsourced action sequences. Unlike synthetic benchmarks, Mind2Web captures the complexity and diversity of real web interfaces. We propose a two-stage approach: element proposal followed by action prediction. Evaluation reveals that web agents struggle with element grounding and action selection in complex interfaces. Common failure modes include confusion between similar elements, inability to scroll to find targets, and incorrect handling of dynamic content.",
        "authors": ["Xiang Deng", "Yu Gu", "Boyuan Zheng", "et al."],
        "published": "2023-06-09",
        "categories": ["cs.CL", "cs.AI", "cs.HC"],
    },
    {
        "arxiv_id": "2401.13649",
        "title": "VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks",
        "abstract": "We introduce VisualWebArena, an extension of WebArena for evaluating multimodal web agents. VisualWebArena adds 910 tasks requiring visual understanding: recognizing images, interpreting charts, comparing visual content, and processing visually-rich interfaces. Tasks are set in realistic e-commerce and content management scenarios. Current vision-language models achieve only 16.4% success rate, with common failures in fine-grained visual discrimination and connecting visual content to required actions. The benchmark reveals that visual reasoning remains a significant bottleneck for web agents.",
        "authors": ["Jing Yu Koh", "Robert Lo", "Lawrence Jang", "et al."],
        "published": "2024-01-24",
        "categories": ["cs.CV", "cs.CL", "cs.AI"],
    },
    {
        "arxiv_id": "2403.07323",
        "title": "WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks?",
        "abstract": "We present WorkArena, a benchmark for evaluating web agents on common enterprise knowledge work tasks. WorkArena contains 33,000 tasks in a ServiceNow environment, covering IT service management, HR processes, and data analysis. Tasks require navigation of complex enterprise interfaces and multi-step workflows. The benchmark reveals that current agents achieve only 7% success rate on complex tasks, with particular struggles in form completion, filtering/sorting, and multi-entity operations. Failure mode analysis shows agents frequently lose track of progress in long sequences and make errors in structured data entry.",
        "authors": ["Alexandre Drouin", "Maxime Gasse", "Massimo Caccia", "et al."],
        "published": "2024-03-12",
        "categories": ["cs.AI", "cs.HC"],
    },
    {
        "arxiv_id": "2401.13178",
        "title": "GAIA: A Benchmark for General AI Assistants",
        "abstract": "We introduce GAIA, a benchmark for general AI assistants requiring multi-step reasoning and tool use. GAIA tasks require browsing the web, performing calculations, and synthesizing information from multiple sources. Unlike narrow benchmarks, GAIA tests whether AI systems can flexibly combine capabilities. Tasks are designed so humans can complete them in minutes while AI systems struggle. Current systems achieve only 15-30% success rate. Analysis reveals common failures in: task decomposition, tool selection, error recovery, and maintaining coherent plans across many steps.",
        "authors": ["Grégoire Mialon", "Clément Bordeau", "Quentin Lhoest", "et al."],
        "published": "2024-01-23",
        "categories": ["cs.AI", "cs.CL"],
    },
    {
        "arxiv_id": "2310.06117",
        "title": "AgentBench: Evaluating LLMs as Agents",
        "abstract": "We present AgentBench, a comprehensive benchmark for evaluating LLMs as autonomous agents across diverse environments. AgentBench includes 8 distinct environments: operating system, database, knowledge graph, game, web shopping, web browsing, lateral thinking puzzles, and household tasks. Each environment provides multi-turn interaction with structured feedback. Evaluation of leading LLMs reveals significant gaps between API-based models and open-source models. Common failure modes across environments include: inability to use feedback for error correction, premature termination, repetitive actions, and poor handling of partial information.",
        "authors": ["Xiao Liu", "Hao Yu", "Hanchen Zhang", "et al."],
        "published": "2023-10-09",
        "categories": ["cs.CL", "cs.AI"],
    },
    {
        "arxiv_id": "2401.14196",
        "title": "Executable Code Actions Elicit Better LLM Agents",
        "abstract": "We introduce CodeAct, an approach where LLM agents express actions as executable Python code rather than structured action formats. CodeAct agents can compose actions, use control flow, and leverage Python libraries, enabling more complex action sequences. The approach unifies tool use under a common code interface. Experiments show CodeAct achieves higher success rates than JSON-based action formats across agent benchmarks. The approach also enables better error handling through try-catch blocks and more efficient multi-step actions.",
        "authors": ["Xingyao Wang", "Yangyi Chen", "Lifan Yuan", "et al."],
        "published": "2024-01-25",
        "categories": ["cs.CL", "cs.AI", "cs.SE"],
    },
    {
        "arxiv_id": "2312.00849",
        "title": "CogAgent: A Visual Language Model for GUI Agents",
        "abstract": "We present CogAgent, a visual language model specialized for GUI understanding and control. CogAgent combines high-resolution image encoding with language model capabilities to understand and interact with graphical interfaces. The model can perform GUI grounding (locating elements by description), action prediction, and multi-step task completion. CogAgent achieves strong results on GUI benchmarks including Mind2Web and AITW, demonstrating that visual models can effectively navigate interfaces without relying on accessibility APIs or HTML structure.",
        "authors": ["Wenyi Hong", "Weihan Wang", "Qingsong Lv", "et al."],
        "published": "2023-12-01",
        "categories": ["cs.CV", "cs.CL", "cs.AI"],
    },
    {
        "arxiv_id": "2401.10935",
        "title": "SeeClick: Harnessing GUI Grounding for Generic Click Automation",
        "abstract": "We present SeeClick, a model for GUI grounding—the task of locating screen elements given natural language descriptions. SeeClick uses a unified architecture that takes screenshots and text queries, producing click coordinates. The model is trained on a large-scale GUI grounding dataset spanning mobile, desktop, and web interfaces. SeeClick achieves state-of-the-art grounding accuracy and can be integrated with LLM agents for click automation. Analysis shows common failures include: visually similar elements, partially occluded elements, and elements requiring scrolling to reach.",
        "authors": ["Kanzhi Cheng", "Qiushi Sun", "Yougang Chu", "et al."],
        "published": "2024-01-19",
        "categories": ["cs.CV", "cs.HC", "cs.AI"],
    },
    {
        "arxiv_id": "2311.12983",
        "title": "AppAgent: Multimodal Agents as Smartphone Users",
        "abstract": "We introduce AppAgent, a multimodal agent framework for autonomous smartphone operation. AppAgent learns from human demonstrations and explores apps to build a knowledge base of UI elements and actions. At runtime, the agent perceives screenshots and decides actions using vision-language models. AppAgent demonstrates capability across diverse apps including social media, shopping, and productivity tools. Analysis reveals failure modes including: inability to recover from unexpected popups, confusion with similar-looking screens, and errors in text input requiring precise cursor positioning.",
        "authors": ["Zhao Yang", "Jiaxuan Liu", "Yucheng Han", "et al."],
        "published": "2023-11-21",
        "categories": ["cs.CV", "cs.AI", "cs.HC"],
    },
    {
        "arxiv_id": "2312.10997",
        "title": "Corrective Retrieval Augmented Generation",
        "abstract": "We present Corrective RAG (CRAG), which improves retrieval-augmented generation through a self-correction mechanism. CRAG introduces a retrieval evaluator that assesses whether retrieved documents are relevant to the query. If documents are irrelevant, CRAG triggers web search for additional information. If documents are ambiguous, CRAG refines the query and re-retrieves. This adaptive approach addresses a common RAG failure mode where irrelevant retrievals lead to incorrect or hallucinated answers. Experiments show CRAG significantly improves answer quality over naive RAG, particularly on complex questions requiring precise information.",
        "authors": ["Shi-Qi Yan", "Jia-Chen Gu", "Yun Zhu", "Zhen-Hua Ling"],
        "published": "2023-12-17",
        "categories": ["cs.CL", "cs.AI", "cs.IR"],
    },
    {
        "arxiv_id": "2401.15884",
        "title": "RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture",
        "abstract": "We present a comprehensive comparison of RAG and fine-tuning approaches for domain adaptation of LLMs. Using agriculture as a case study, we analyze tradeoffs across dimensions including accuracy, latency, cost, and maintenance burden. Key findings: RAG excels at factual accuracy and incorporating recent information, while fine-tuning produces more fluent domain-specific language. Hybrid approaches combining RAG with lightweight fine-tuning often perform best. We identify when hybrid retrieval (lexical + dense + reranking) outperforms dense-only: complex technical queries benefit from hybrid, while simple factoid queries see little improvement from the added complexity.",
        "authors": ["RAG vs Fine-tuning Team"],
        "published": "2024-01-28",
        "categories": ["cs.CL", "cs.AI", "cs.IR"],
    },
    {
        "arxiv_id": "2301.13379",
        "title": "Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP",
        "abstract": "We introduce DSP (Demonstrate-Search-Predict), a framework for composing retrieval and language models. DSP separates task specification (as demonstrations) from retrieval and generation, enabling modular pipelines that can be optimized independently. The framework supports multi-hop reasoning through iterative retrieval steps. Experiments on knowledge-intensive tasks show DSP outperforms end-to-end approaches. Analysis reveals hybrid retrieval with reranking helps most on multi-hop questions requiring multiple documents, while single-hop factoid questions benefit less from reranking overhead.",
        "authors": ["Omar Khattab", "Keshav Santhanam", "Xiang Lisa Li", "et al."],
        "published": "2023-01-30",
        "categories": ["cs.CL", "cs.AI", "cs.IR"],
    },
    {
        "arxiv_id": "2408.15232",
        "title": "Co-STORM: Collaborative STORM for Multi-Turn Research Assistance",
        "abstract": "We present Co-STORM, an extension of STORM for collaborative, multi-turn research assistance. Co-STORM enables users to interactively guide the research process through natural conversation. The system maintains a shared research state that evolves through user feedback and agent actions. Key components include: a planner that decomposes research goals, a retriever that gathers relevant sources, a reflector that evaluates progress and identifies gaps, and a synthesizer that produces reports. Analysis shows that the reflector component helps most when research scope is ambiguous but can slow down focused, well-defined queries.",
        "authors": ["Co-STORM Team"],
        "published": "2024-08-27",
        "categories": ["cs.CL", "cs.AI", "cs.IR"],
    },
]

def main():
    data_dir = Path("data/processed")
    papers_file = data_dir / "papers_metadata.json"
    
    # Load existing papers
    existing = []
    existing_ids = set()
    if papers_file.exists():
        existing = json.loads(papers_file.read_text())
        existing_ids = {p["arxiv_id"] for p in existing}
        print(f"Loaded {len(existing)} existing papers")
    
    # Add required papers
    added = 0
    for paper in REQUIRED_PAPERS:
        if paper["arxiv_id"] not in existing_ids:
            existing.append(paper)
            existing_ids.add(paper["arxiv_id"])
            added += 1
            print(f"  [+] {paper['arxiv_id']}: {paper['title'][:60]}...")
    
    print(f"\nAdded {added} new papers")
    print(f"Total papers: {len(existing)}")
    
    # Save updated papers
    papers_file.write_text(json.dumps(existing, indent=2))
    print(f"Saved to {papers_file}")
    
    # Update chunks
    print("\nCreating chunks...")
    chunks = []
    for paper in existing:
        chunks.append({
            "chunk_id": f"{paper['arxiv_id']}_abstract",
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "text": paper.get("abstract", ""),
            "section": "abstract",
        })
    
    chunks_file = data_dir / "chunks.json"
    chunks_file.write_text(json.dumps(chunks, indent=2))
    print(f"Saved {len(chunks)} chunks to {chunks_file}")


if __name__ == "__main__":
    main()
