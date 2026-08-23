from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

_tavily = TavilySearch(max_results=5)
_search_count = {"n": 0}
MAX_SEARCHES = 2


@tool
def search_web(query: str) -> str:
    """Search the web for information. Only pass a search query string."""
    _search_count["n"] += 1
    if _search_count["n"] > MAX_SEARCHES:
        return """
                ERROR: TOOL DISABLED. No further searches possible.
        """

    result = _tavily.invoke({"query": query})
    return str(result)[:2000]  # обрезаем, чтобы не забивать контекст


template = """Answer the following questions as best you can. You have access to the following tools:
{tools}
CRITICAL RULES:
- Only use tool names from this exact list: [{tool_names}]. Never invent a tool name.
- Do NOT call search_web more than once with a similar query. If you already have enough information from previous Observations, go straight to Final Answer.
- You do not need a separate tool to summarize or extract details — just write them yourself in the Final Answer using the Observation data you already have.
IMPORTANT: You must ALWAYS follow every Thought with an Action and Action Input on the next lines. Never write a Thought without immediately following it with an Action.
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)
tools = [search_web]

llm = ChatOllama(model="llama3.2", temperature=0)
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="force",
    return_intermediate_steps=True,
)

result = agent_executor.invoke({"input": """
    search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details
    - When searching, use specific queries that target individual job postings, not aggregator pages (avoid terms like 'jobs in' that return listing pages).
    """})

if "stopped due to" in result["output"]:
    steps = result.get("intermediate_steps", [])
    # Отбираем только шаги с реальными данными, а не с заглушкой об ошибке
    valid_observations = [
        str(obs) for _, obs in steps if "TOOL DISABLED" not in str(obs)
    ]
    if valid_observations:
        combined = "\n\n---\n\n".join(valid_observations)[:3000]
        fallback_prompt = (
            f"Based on this search data, list 3 AI Engineer job postings with details "
            f"(title, location, salary if available, company):\n\n{combined}"
        )
        fallback_response = llm.invoke(fallback_prompt)
        print(fallback_response.content)
    else:
        print("No valid search data was collected.")
else:
    print(result["output"])

print("\n--- FINAL ANSWER ---")
