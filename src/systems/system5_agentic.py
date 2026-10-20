"""
System 5: Agentic RAG with LangGraph + memory

Features:
1. AgentState with memory (failed_tools, search_queries)
2. System message with clear tool priorities
3. Does not repeat tools that already failed
4. Finishes sooner (2-3 tool calls instead of 6)
5. Tools live in agent_tools.py (single source of truth)
"""

import time
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from src.llm.llm_client import glm_endpoint

# Import tools (single source of truth)
from src.tools.agent_tools import TOOLS


def _make_llm() -> ChatOpenAI:
    """Chat model on the self-hosted endpoint configured through GLM_* variables."""
    return ChatOpenAI(**glm_endpoint(), temperature=0.0)


# ============================================================================
# STATE DEFINITION WITH MEMORY
# ============================================================================

class AgentState(TypedDict):
    """State with memory - the agent learns from past actions."""
    messages: Annotated[List[BaseMessage], add_messages]
    iterations: int
    failed_tools: List[str]  # tools that already failed
    search_queries_used: List[str]  # queries already used
    found_answer: bool  # whether the answer was already found


# ============================================================================
# AGENT NODES
# ============================================================================

def should_continue(state: AgentState) -> str:
    """Decides whether to continue or stop."""
    MAX_ITERATIONS = 3

    messages = state["messages"]
    last_message = messages[-1]

    # Safety: force a final answer after max iterations
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return "final_answer"

    # Tool calls requested and still under the limit: continue
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        if state.get("iterations", 0) < MAX_ITERATIONS:
            return "tools"
        return "final_answer"

    # Otherwise end
    return "end"


def call_model(state: AgentState) -> Dict:
    """Calls the LLM with the memory context."""

    messages = list(state["messages"])

    # Inject memory on every iteration
    failed_tools = state.get("failed_tools", [])
    search_count = len(state.get("search_queries_used", []))
    iteration = state.get("iterations", 0)

    if iteration > 0:
        memory_context = f"""[Memory - Iteration {iteration}]
Tools that failed: {', '.join(failed_tools) if failed_tools else 'none'}
Searches done: {search_count}
Reminder: Prefer search() over get_financial_facts()"""

        messages.append(SystemMessage(content=memory_context))

    llm = _make_llm()

    # Bind tools (TOOLS from agent_tools.py)
    llm_with_tools = llm.bind_tools(TOOLS)

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1
    }


def update_memory(state: AgentState) -> Dict:
    """Updates the memory from tool results.

    Runs AFTER tool execution to track which tools failed."""

    messages = state["messages"]
    failed_tools = list(state.get("failed_tools", []))
    search_queries = list(state.get("search_queries_used", []))
    found_answer = state.get("found_answer", False)

    # Inspect the latest messages for tool results
    for msg in reversed(messages[-5:]):  # last 5 messages

        # ToolMessage = tool result
        if isinstance(msg, ToolMessage):
            content = msg.content
            tool_name = msg.name if hasattr(msg, 'name') else 'unknown'

            # If the tool failed, record it
            if "No facts found" in content or "Error:" in content:
                tool_key = f"{tool_name}"
                if tool_key not in failed_tools:
                    failed_tools.append(tool_key)

            # If search found information, record it
            if tool_name == 'search' and "Found" in content:
                # Track the query (look in the previous AIMessage)
                for prev_msg in reversed(messages):
                    if isinstance(prev_msg, AIMessage) and hasattr(prev_msg, 'tool_calls'):
                        for tc in prev_msg.tool_calls:
                            if tc['name'] == 'search':
                                query = tc['args'].get('query', '')
                                if query and query not in search_queries:
                                    search_queries.append(query)
                        break

                # Financial terms in the results suggest the answer is already there
                if "Found" in content and any(word in content.lower() for word in ['billion', 'million', 'revenue', 'sales']):
                    found_answer = True

    return {
        "failed_tools": failed_tools,
        "search_queries_used": search_queries,
        "found_answer": found_answer
    }


def generate_final_answer(state: AgentState) -> Dict:
    """Forces the LLM to write a final answer from the tool results.

    Runs when MAX_ITERATIONS is reached, to avoid empty answers."""

    messages = list(state["messages"])

    # Append a message that forces an answer
    force_answer_msg = SystemMessage(content="""Based on the tool results above, provide your FINAL ANSWER now.

If you found the information, state it clearly and concisely.
If you did NOT find the information, say "I don't have enough information to answer this question."

DO NOT make tool calls. Just answer with what you found.""")

    messages.append(force_answer_msg)

    # LLM WITHOUT tools, to force a direct answer
    llm = _make_llm()

    response = llm.invoke(messages)

    return {"messages": [response]}


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_agent_graph():
    """Creates the LangGraph agent with memory."""

    # Load .env
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / ".env")

    # Pre-load BGE-M3
    print("  Pre-loading BGE-M3...")
    from FlagEmbedding import BGEM3FlagModel
    _ = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    print("    BGE-M3 loaded")

    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(TOOLS))  # TOOLS from agent_tools.py
    workflow.add_node("memory", update_memory)
    workflow.add_node("final_answer", generate_final_answer)  # forces a final answer

    # Entry
    workflow.set_entry_point("agent")

    # Routing
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "final_answer": "final_answer",  # when MAX_ITERATIONS is reached
            "end": END
        }
    )

    # Tools → memory → agent (loop)
    workflow.add_edge("tools", "memory")
    workflow.add_edge("memory", "agent")

    # Final answer → END (always ends after generating the answer)
    workflow.add_edge("final_answer", END)

    return workflow.compile()


# ============================================================================
# SYSTEM CLASS
# ============================================================================

class System5_Agentic:
    """System 5: agentic RAG with memory."""

    def __init__(self):
        print("\n[System 5: Agentic RAG with memory]")
        print("-" * 60)

        self.agent = create_agent_graph()

        print("  LangGraph agent created")
        print("  Tools: search, calculator, get_financial_facts")
        print("  Memory: failed_tools, search_queries, found_answer")
        print("-" * 60 + "\n")

    def answer(self, query: str, include_trace: bool = False) -> Dict:
        """Answers a question with the agent."""

        start_time = time.time()

        system_msg = """You are a financial analyst. Answer efficiently.

TOOL PRIORITY:
1. get_financial_facts(company, metric) - Structured XBRL data (BEST for specific metrics)
2. search(query) - Full-text search in 10-K/10-Q (for context & details)
3. calculator(expression) - Math operations

CRITICAL RULES:
- Start with search() for financial questions
- Do NOT retry tools that failed
- Do NOT repeat similar queries
- Answer IMMEDIATELY when you find the info (2-3 tools max)
- Quality over quantity of tool calls"""

        # Initial state with empty memory
        initial_state = {
            "messages": [
                SystemMessage(content=system_msg),
                HumanMessage(content=query)
            ],
            "iterations": 0,
            "failed_tools": [],
            "search_queries_used": [],
            "found_answer": False
        }

        result = self.agent.invoke(initial_state)

        # Extract answer
        messages = result["messages"]
        final_message = messages[-1]

        if isinstance(final_message, AIMessage):
            answer = final_message.content
        else:
            answer = str(final_message)

        end_time = time.time()

        # Count tool calls
        tool_calls = sum(
            1 for msg in messages
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls
        )

        # Extract chunk texts from the ToolMessages (for RAGAS)
        chunk_texts = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                if hasattr(msg, 'name') and msg.name == 'search':
                    if msg.content and len(msg.content) > 50:
                        chunk_texts.append(msg.content)

        result_dict = {
            'query': query,
            'answer': answer,
            'tool_calls': tool_calls,
            'time_seconds': end_time - start_time,
            'failed_tools': result.get('failed_tools', []),
            'searches': len(result.get('search_queries_used', [])),
            'chunk_texts': chunk_texts  # used by RAGAS (same as Systems 1-3)
        }

        if include_trace:
            result_dict['messages'] = [
                {
                    'type': type(msg).__name__,
                    'content': msg.content if hasattr(msg, 'content') else str(msg),
                    'tool_calls': msg.tool_calls if hasattr(msg, 'tool_calls') and msg.tool_calls else None,
                    'name': msg.name if hasattr(msg, 'name') else None
                }
                for msg in messages
            ]

        return result_dict


def main():
    """System 5 smoke test (run with `python -m src.systems.system5_agentic`)."""
    print("\n" + "="*80)
    print("SYSTEM 5 - TEST")
    print("="*80)

    query = "What was Apple's total revenue in fiscal year 2023?"

    print(f"\nQuery: {query}")
    print("="*80)

    system5 = System5_Agentic()

    result = system5.answer(query, include_trace=True)

    print(f"\nAnswer: {result['answer']}")
    print(f"\nMetrics:")
    print(f"  Tool calls: {result['tool_calls']}")
    print(f"  Searches: {result['searches']}")
    print(f"  Failed tools: {result['failed_tools']}")
    print(f"  Time: {result['time_seconds']:.2f}s")

    print("\n" + "="*80)
    print("Test completed")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
