#!/usr/bin/env python3
"""
Cost estimation (tokens, LLM calls, time, USD) for the 5 systems over the 35
validated questions, WITHOUT re-running any system.

Common method for all systems:
- The exact prompt each system sends to the LLM is rebuilt from the templates in
  the source code (src/systems/*) and from the retrieved texts stored in the
  results (retrieved_contexts).
- Every system is tokenized with o200k_base (gpt-4o-mini tokenizer).
- The chat-format overhead is added (3 tokens per message + 3 closing tokens).
- The USD cost uses the gpt-4o-mini price ($0.15 / 1M input, $0.60 / 1M output)
  as an equivalent cost for every system.

Assumptions per system:
- S1, S2, S3: 1 call. Prompt = system message + per-document headers +
  full text of the 8 chunks + question.
- S4: 1 call. Chunks truncated to 1000 characters + graph section
  (5 entities + 5 relations), estimated with the average size of the real
  entities and relations of the knowledge graph.
- S5 (LangGraph, MAX_ITERATIONS=3): calls = 1 (no tools), 2 (1 tool call) or
  4 (2-3 tool calls: 3 from the agent + final_answer). Every call resends the
  whole history (system, question, tool calls, tool outputs, memory) plus the
  schemas of the 3 tools. search outputs are exact (retrieved_contexts);
  get_financial_facts / calculator outputs that were not saved are approximated.
  Hidden GLM reasoning tokens are not observable and are not included.

Outputs:
- data/evaluation/cost_estimates.json
- data/evaluation/cost_estimates_per_question.csv

Usage: python3 estimate_costs.py
"""

import ast
import csv
import json
import re
import statistics as st
from pathlib import Path

import tiktoken

ROOT = Path(__file__).parent
EVAL_DIR = ROOT / "data/evaluation"
KG_DIR = ROOT / "data/knowledge_graph"

VALID_IDS = {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24,
             25, 26, 27, 28, 30, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60}

PRICE_IN = 0.15 / 1_000_000   # gpt-4o-mini input
PRICE_OUT = 0.60 / 1_000_000  # gpt-4o-mini output
CONTEXT_LIMIT = 128_000       # gpt-4o-mini context window

MSG_OVERHEAD = 3   # tokens per message in chat format
REPLY_PRIMING = 3  # closing tokens per request

SYSTEM_NAMES = {
    1: "S1 BM25",
    2: "S2 Vector",
    3: "S3 Hybrid",
    4: "S4 Hybrid+Graph",
    5: "S5 Agentic",
}

enc = tiktoken.get_encoding("o200k_base")


def ntok(text: str) -> int:
    return len(enc.encode(text or "", disallowed_special=()))


def chat_input(messages: list) -> int:
    """Input tokens of a chat request (list of strings, one per message)."""
    return sum(ntok(m) + MSG_OVERHEAD for m in messages) + REPLY_PRIMING


# ============================================================================
# Templates (copied from each system's code)
# ============================================================================

SYSTEM_MSG_S123 = """You are a financial analyst assistant. Answer questions based ONLY on the provided documents.

Rules:
- Cite sources with [Document N] references
- If information is not in the documents, say so
- For numerical answers, include units and time periods
- Be concise and factual"""

SYSTEM_MSG_S4 = "You are a financial analyst. Answer questions based on SEC filings data."

SYSTEM_MSG_S5 = """You are a financial analyst. Answer efficiently.

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

FORCE_ANSWER_S5 = """Based on the tool results above, provide your FINAL ANSWER now.

If you found the information, state it clearly and concisely.
If you did NOT find the information, say "I don't have enough information to answer this question."

DO NOT make tool calls. Just answer with what you found."""

RELEVANCE_LABEL = {1: "BM25=0.0000", 2: "Similarity=0.0000", 3: "RRF=0.0000"}

HEADER_RE = re.compile(r"\[Company: (?P<company>[^|]+?) \| Document: (?P<form>\S+) (?P<period>\S+)")


def parse_header(chunk_text: str) -> dict:
    m = HEADER_RE.search(chunk_text[:400])
    if not m:
        return {"company": "UNKNOWN", "form": "UNKNOWN", "period": "UNKNOWN"}
    return m.groupdict()


# ============================================================================
# S1, S2, S3
# ============================================================================

def estimate_s123(system_id: int, question: str, contexts: list, answer: str) -> dict:
    if contexts:
        parts = [f"Retrieved {len(contexts)} relevant documents:\n"]
        for i, text in enumerate(contexts, 1):
            h = parse_header(text)
            parts.append(f"\n[Document {i}]")
            parts.append(f"Source: {h['company']} {h['form']} ({h['period']})")
            parts.append("Section: Item N/A")
            parts.append(f"Relevance: {RELEVANCE_LABEL[system_id]}\n")
            parts.append(text)
            parts.append("\n" + "-" * 80)
        context = "\n".join(parts)
    else:
        context = "No relevant documents found."

    prompt = f"""Based on the following documents, answer the question.

{context}

Question: {question}

Answer:"""

    tin = chat_input([SYSTEM_MSG_S123, prompt])
    return {"input_tokens": tin, "output_tokens": ntok(answer), "llm_calls": 1}


# ============================================================================
# S4
# ============================================================================

def kg_line_averages() -> tuple:
    """Average tokens of an entity line and of a relation line of the graph."""
    ent = [json.loads(l) for l in (KG_DIR / "entities.jsonl").open()]
    rel = [json.loads(l) for l in (KG_DIR / "relations.jsonl").open()]
    ent_avg = st.mean(ntok(f"  - {e['nombre']} ({e['tipo']})") for e in ent)
    rel_avg = st.mean(ntok(f"  {r['origen']} --[{r['relacion']}]--> {r['destino']}") for r in rel)
    return ent_avg, rel_avg


def estimate_s4(question: str, contexts: list, answer: str, n_entities: int,
                ent_avg: float, rel_avg: float) -> dict:
    chunk_texts = []
    for i, text in enumerate(contexts, 1):
        h = parse_header(text)
        chunk_texts.append(f"[Chunk {i}] {h['company']} {h['form']}\n{text[:1000]}")
    chunks_text = "\n\n".join(chunk_texts)

    prompt = f"""Answer the following question based on the provided SEC filing excerpts.

DOCUMENT EXCERPTS:
{chunks_text}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based on the document excerpts above
2. Be precise and cite specific numbers/facts when available
3. If the answer is partially in the documents, provide what you can find
4. Only say "I don't have enough information" if the documents contain NO relevant information

ANSWER:"""

    tin = chat_input([SYSTEM_MSG_S4, prompt])

    # Graph section: only added when there are entities (len(graph) > 2)
    n_ent = min(n_entities or 0, 5)
    if n_ent > 0:
        graph_tokens = (
            ntok("\n\nADDITIONAL CONTEXT (Entity Relationships):\nENTITIES FOUND:")
            + n_ent * ent_avg
            + ntok("\nRELATED ENTITIES (10 total):")
            + 5 * rel_avg
        )
        tin += round(graph_tokens)

    return {"input_tokens": tin, "output_tokens": ntok(answer), "llm_calls": 1}


# ============================================================================
# S5
# ============================================================================

def tool_schemas_tokens() -> int:
    """Tokens of the schemas of the 3 tools sent with bind_tools()."""
    src = (ROOT / "src/tools/agent_tools.py").read_text()
    tree = ast.parse(src)
    tools = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(d, ast.Name) and d.id == "tool" for d in node.decorator_list
        ):
            props = {a.arg: {"type": "string"} for a in node.args.args}
            tools.append({
                "type": "function",
                "function": {
                    "name": node.name,
                    "description": ast.get_docstring(node) or "",
                    "parameters": {"type": "object", "properties": props,
                                   "required": [node.args.args[0].arg]},
                },
            })
    return ntok(json.dumps(tools))


# Tool outputs that were not saved
FACTS_FAIL_MSG = "No facts found for AAPL - Revenues"
UNSAVED_TOOL_OUTPUT_TOKENS = 150  # successful get_financial_facts (3 JSON periods) or calculator


def estimate_s5(question: str, contexts: list, answer: str, tool_calls: int,
                failed_tools: list, schemas: int) -> dict:
    search_call = json.dumps({"name": "search", "arguments": {"query": question}})
    facts_call = json.dumps({"name": "get_financial_facts",
                             "arguments": {"company": "AAPL", "metric": "Revenues"}})

    # Turns with executed tools (the 3rd tool call is not executed: MAX_ITERATIONS=3)
    executed_turns = min(tool_calls, 2)

    # Tool outputs per turn: search first (the system prompt says so), then facts
    outputs = [ntok(c) for c in contexts]
    calls = [ntok(search_call)] * len(contexts)
    if "get_financial_facts" in failed_tools:
        outputs.append(ntok(FACTS_FAIL_MSG))
        calls.append(ntok(facts_call))
    while len(outputs) < executed_turns:
        outputs.append(UNSAVED_TOOL_OUTPUT_TOKENS)
        calls.append(ntok(facts_call))

    turns = [{"call_tokens": 0, "output_tokens": 0, "n_msgs": 0} for _ in range(executed_turns)]
    for i, (c, o) in enumerate(zip(calls, outputs)):
        t = turns[min(i, executed_turns - 1)] if executed_turns else None
        if t is None:
            break
        t["call_tokens"] += c
        t["output_tokens"] += o
        t["n_msgs"] += 1

    answer_tok = ntok(answer)
    base = ntok(SYSTEM_MSG_S5) + ntok(question) + 2 * MSG_OVERHEAD
    memory_msg = ntok(f"""[Memory - Iteration 1]
Tools that failed: {', '.join(failed_tools) if failed_tools else 'none'}
Searches done: 1
Reminder: Prefer search() over get_financial_facts()""") + MSG_OVERHEAD

    total_in, total_out, llm_calls = 0, 0, 0
    history = 0  # accumulated tokens of tool calls + tool outputs

    def agent_call(iteration: int) -> int:
        mem = memory_msg if iteration > 0 else 0
        return base + history + mem + schemas + REPLY_PRIMING

    if tool_calls == 0:
        total_in += agent_call(0)
        total_out += answer_tok
        llm_calls = 1
    else:
        # Turns with executed tools
        for it, t in enumerate(turns):
            total_in += agent_call(it)
            total_out += t["call_tokens"]
            llm_calls += 1
            history += t["call_tokens"] + MSG_OVERHEAD
            history += t["output_tokens"] + t["n_msgs"] * MSG_OVERHEAD

        if tool_calls == 1:
            # The agent answers directly on the 2nd call
            total_in += agent_call(1)
            total_out += answer_tok
            llm_calls += 1
        else:
            # 3rd agent call (unexecuted tool call or answer) + final_answer
            total_in += agent_call(2)
            third_out = ntok(search_call) if tool_calls >= 3 else answer_tok
            total_out += third_out
            history += third_out + MSG_OVERHEAD
            llm_calls += 1

            total_in += base + history + memory_msg + ntok(FORCE_ANSWER_S5) + MSG_OVERHEAD + REPLY_PRIMING
            total_out += answer_tok
            llm_calls += 1

    return {"input_tokens": total_in, "output_tokens": total_out, "llm_calls": llm_calls}


# ============================================================================
# Main
# ============================================================================

def load_results() -> dict:
    """Returns {system_id: {question_id: row}} from the final sources used in RAGAS."""
    by_sys = {s: {} for s in SYSTEM_NAMES}
    for line in (EVAL_DIR / "results_60q.jsonl").open():
        r = json.loads(line)
        if r["system_id"] in (1, 2, 3) and r["question_id"] in VALID_IDS:
            by_sys[r["system_id"]][r["question_id"]] = r
    for sid, fname in [(4, "results_s4.jsonl"),
                       (5, "results_s5.jsonl")]:
        for line in (EVAL_DIR / fname).open():
            r = json.loads(line)
            by_sys[sid][r["question_id"]] = r
    return by_sys


def summarize(rows: list) -> dict:
    n = len(rows)
    tin = [r["input_tokens"] for r in rows]
    tout = [r["output_tokens"] for r in rows]
    cost = [r["cost_usd"] for r in rows]
    return {
        "n_questions": n,
        "mean_input_tokens": round(st.mean(tin)),
        "mean_output_tokens": round(st.mean(tout)),
        "mean_total_tokens": round(st.mean(t + o for t, o in zip(tin, tout))),
        "median_total_tokens": round(st.median(t + o for t, o in zip(tin, tout))),
        "total_tokens": sum(tin) + sum(tout),
        "mean_llm_calls": round(st.mean(r["llm_calls"] for r in rows), 2),
        "mean_latency_s": round(st.mean(r["time_seconds"] for r in rows), 2),
        "mean_cost_usd": round(st.mean(cost), 6),
        "total_cost_usd": round(sum(cost), 4),
        "cost_per_1000_queries_usd": round(st.mean(cost) * 1000, 2),
    }


def main():
    by_sys = load_results()
    ent_avg, rel_avg = kg_line_averages()
    schemas = tool_schemas_tokens()

    per_question = []
    for sid, rows in by_sys.items():
        for qid, r in sorted(rows.items()):
            contexts = r.get("retrieved_contexts") or []
            answer = r.get("answer") or ""
            if sid in (1, 2, 3):
                est = estimate_s123(sid, r["question"], contexts, answer)
            elif sid == 4:
                est = estimate_s4(r["question"], contexts, answer,
                                  r.get("graph_entities", 0), ent_avg, rel_avg)
            else:
                est = estimate_s5(r["question"], contexts, answer, r.get("tool_calls", 0),
                                  r.get("failed_tools", []), schemas)

            overflow = est["input_tokens"] > CONTEXT_LIMIT
            answered = bool(r.get("answer"))
            per_question.append({
                "system_id": sid,
                "system": SYSTEM_NAMES[sid],
                "question_id": qid,
                "category": r.get("category"),
                "answered": answered,
                "context_overflow": overflow,
                "input_tokens": est["input_tokens"],
                "output_tokens": est["output_tokens"],
                "total_tokens": est["input_tokens"] + est["output_tokens"],
                "llm_calls": est["llm_calls"],
                "tool_calls": r.get("tool_calls", 0) if sid == 5 else 0,
                "time_seconds": round(r.get("time_seconds") or 0, 2),
                "cost_usd": est["input_tokens"] * PRICE_IN + est["output_tokens"] * PRICE_OUT,
            })

    # Common set: questions answered by all 5 systems (same comparison basis)
    answered_ids = [
        {q["question_id"] for q in per_question if q["system_id"] == sid and q["answered"]}
        for sid in SYSTEM_NAMES
    ]
    common = set.intersection(*answered_ids)

    summary_all, summary_common = {}, {}
    for sid, name in SYSTEM_NAMES.items():
        rows = [q for q in per_question if q["system_id"] == sid]
        summary_all[name] = summarize(rows)
        summary_all[name]["unanswered"] = sum(not q["answered"] for q in rows)
        summary_all[name]["context_overflows"] = sum(q["context_overflow"] for q in rows)
        summary_common[name] = summarize([q for q in rows if q["question_id"] in common])

    output = {
        "method": {
            "tokenizer": "o200k_base (gpt-4o-mini), applied to all systems",
            "pricing": "gpt-4o-mini: $0.15/1M input, $0.60/1M output (equivalent cost for all 5)",
            "chat_overhead": f"{MSG_OVERHEAD} tokens/message + {REPLY_PRIMING} per request",
            "s4_graph_line_avg_tokens": {"entity": round(ent_avg, 1), "relation": round(rel_avg, 1)},
            "s5_tool_schema_tokens": schemas,
            "s5_unsaved_tool_output_tokens": UNSAVED_TOOL_OUTPUT_TOKENS,
            "notes": [
                "No system was re-run; prompts are rebuilt from the code and the saved results.",
                "S5 ran on a self-hosted GLM-5.3-Flash endpoint (real cost $0); the USD figure is the gpt-4o-mini equivalent.",
                "S5 excludes hidden GLM reasoning tokens (not observable in the results).",
                "Overflows: the estimated prompt exceeds 128k tokens; the API rejects the request (no answer).",
            ],
        },
        "common_question_ids": sorted(common),
        "summary_common_set": summary_common,
        "summary_all_35": summary_all,
    }

    out_json = EVAL_DIR / "cost_estimates.json"
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    out_csv = EVAL_DIR / "cost_estimates_per_question.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_question[0].keys()))
        writer.writeheader()
        writer.writerows(per_question)

    # Console report
    print(f"Common set: {len(common)} questions answered by all 5 systems\n")
    header = f"{'System':<17}{'In':>9}{'Out':>7}{'Total':>9}{'Calls':>7}{'Lat(s)':>8}{'$/1k q':>9}"
    for title, summary in [("COMMON SET", summary_common), ("ALL (35)", summary_all)]:
        print(title)
        print(header)
        for name, s in summary.items():
            print(f"{name:<17}{s['mean_input_tokens']:>9}{s['mean_output_tokens']:>7}"
                  f"{s['mean_total_tokens']:>9}{s['mean_llm_calls']:>7}{s['mean_latency_s']:>8}"
                  f"{s['cost_per_1000_queries_usd']:>9}")
        print()
    print(f"Saved: {out_json}\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
