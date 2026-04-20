from langchain_core.tools import tool


@tool
def mock_searxng_search(query: str):
    """Return hardcoded 'recent headlines' based on keyword matches in the query."""
    query = query.lower()

    if "crypto" in query:
        return ["Bitcoin hits new all-time high amid ETF approvals"]
    elif "ai" in query:
        return ["OpenAI releases powerful new AI model"]
    elif "market" in query:
        return ["Stock market surges after interest rate cuts"]
    else:
        return ["Global economy facing uncertainty"]