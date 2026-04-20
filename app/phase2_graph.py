from __future__ import annotations

from typing import List, TypedDict

from langgraph.graph import StateGraph

try:
    from app.llm import call_llm_json
    from app.tools import mock_searxng_search
except ImportError:
    from llm import call_llm_json
    from tools import mock_searxng_search


class State(TypedDict, total=False):
    persona: str
    query: str
    search_results: List[str]
    output: dict


def decide_search(state: State) -> State:
    prompt = f"""
You are an AI with persona: {state['persona']}
Decide a trending topic and return ONLY a short search query.
""".strip()
    try:
        from app.llm import call_llm
    except ImportError:
        from llm import call_llm

    query_str = call_llm(prompt).strip()
    return {"query": query_str}


def web_search(state: State) -> State:
    results = mock_searxng_search.invoke({"query": state["query"]})
    return {"search_results": results}


def draft_post(state: State) -> State:
    prompt = f"""
Persona: {state['persona']}
Context: {state['search_results']}

Generate STRICT JSON:
{{
  "bot_id": "...",
  "topic": "...",
  "post_content": "..."
}}
""".strip()
    parsed = call_llm_json(prompt)
    return {"output": parsed}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("decide_search", decide_search)
    builder.add_node("web_search", web_search)
    builder.add_node("draft_post", draft_post)

    builder.set_entry_point("decide_search")
    builder.add_edge("decide_search", "web_search")
    builder.add_edge("web_search", "draft_post")

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke({"persona": "Bot A (Tech Maximalist)"})
    print(result["output"])