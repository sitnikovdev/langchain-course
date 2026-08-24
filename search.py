from rich.console import Console
from rich.markdown import Markdown
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.callbacks import BaseCallbackHandler


class CleanConsoleCallback(BaseCallbackHandler):
    def on_agent_action(self, action, **kwargs):
        if not action.log.strip():
            return  # пропускаем пустые/служебные шаги
        print(f"\n{'='*60}")
        print(f"💭 Thought/Action log:\n{action.log.strip()}")
        print(f"{'='*60}\n")

    def on_agent_finish(self, finish, **kwargs):
        print(f"\n{'='*60}")
        print("✅ FINAL ANSWER")
        print(f"{'='*60}\n")


_tavily = TavilySearch(max_results=5)
_search_count = {"n": 0}
MAX_SEARCHES = 3


@tool
def search_web(query: str) -> str:
    """Search the web for information. Only pass a search query string."""
    _search_count["n"] += 1
    if _search_count["n"] > MAX_SEARCHES:
        msg = "ERROR: TOOL DISABLED. No further searches possible."
        print(f"📥 Observation (preview):\n{msg}\n")
        return msg

    result = _tavily.invoke({"query": query})
    output = str(result)[:4000]
    print(f"📥 Observation (preview):\n{output[:300]}...\n")
    return output


template = """Answer the following questions as best you can. You have access to the following tools:
{tools}
CRITICAL RULES:
- Only use tool names from this exact list: [{tool_names}]. Never invent a tool name.
- Do NOT call search_web more than once with a similar query. If you already have enough information from previous Observations, go straight to Final Answer.
- When writing your Final Answer, always extract and include ALL available details from the Observation data — especially URLs/links, company names, and salary figures. Do not omit a field just because it takes extra effort to find in the raw data; scan the full Observation text for it.
- You do not need a separate tool to summarize or extract details — just write them yourself in the Final Answer using the Observation data you already have.
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

llm = ChatOllama(model="qwen3:8b", temperature=0)
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="force",
    return_intermediate_steps=True,
    callbacks=[CleanConsoleCallback()],
)

result = agent_executor.invoke({"input": """
    Search for 3 job postings for an AI Engineer using LangChain in the Bay Area on LinkedIn.
    - When searching, use specific queries that target individual job postings, not aggregator pages (avoid terms like 'jobs in' that return listing pages).
    - Use up to 3 different search queries with clearly different phrasing to find distinct job postings.
    - Before listing postings, check if two results describe the same role — if so, merge them, don't list duplicates.

    CRITICAL: When you are ready to give your answer, you MUST start that section with the exact
    literal text "Final Answer:" on its own — this is a strict format requirement, not optional.
    Do NOT just start writing the list directly after a Thought.

    For your Final Answer, format EACH job posting using EXACTLY this structure:

    N. [Job Title]
       • Company: [company name]
       • Location: [city/area]
       • Salary: [salary range or "Not specified"]
       • Description: [1-2 sentence summary]
       • Link: [full URL from the search result]
    """})

console = Console()
console.print(Markdown(result["output"]))

print("\n" + "=" * 60)
print("📥 ALL OBSERVATIONS (post-hoc)")
print("=" * 60)
for i, (action, observation) in enumerate(
    [(a, o) for a, o in result.get("intermediate_steps", []) if a.log.strip()], 1
):
    print(f"\n[{i}] Query: {action.tool_input}")
    print(f"    Result: {str(observation)[:300]}...")
