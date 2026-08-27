from rich.console import Console
from rich.markdown import Markdown
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_tavily import TavilySearch
from langchain_core.callbacks import BaseCallbackHandler


class CleanConsoleCallback(BaseCallbackHandler):
    """Лёгкий live-индикатор прогресса. Полная история шагов печатается один раз
    в конце, из intermediate_steps — без дублирования."""

    def on_agent_action(self, action, **kwargs):
        if not action.log.strip():
            return
        print(f"🔧 {action.tool}({action.tool_input}) ...")

    def on_agent_finish(self, finish, **kwargs):
        print("✅ Готов финальный ответ\n")


_tavily = TavilySearch(max_results=5)


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y


@tool
def search_web(query: str) -> str:
    """Search the web for information. Pass a focused search query string."""
    result = _tavily.invoke({"query": query})
    items = result.get("results", []) if isinstance(result, dict) else []
    cleaned = [
        {"title": r.get("title"), "content": r.get("content", "")[:500]}
        for r in items[:3]
    ]
    return str(cleaned)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. When comparing data (e.g. weather in "
            "two cities), search for each location separately, then give a short, "
            "direct comparison in the requested units. Do not include raw search "
            "artifacts in your final answer — summarize in plain language.",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

tools = [search_web, multiply]
llm = ChatOllama(model="qwen3:8b", temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=6,
    early_stopping_method="force",
    return_intermediate_steps=True,
    callbacks=[CleanConsoleCallback()],
)

result = agent_executor.invoke(
    {
        "input": "What is the weather in Saransk right now? Compare it with San Francisco. "
        "Output temperatures in Celsius.",
    }
)

console = Console()
console.print(Markdown(result["output"]))

steps = [(a, o) for a, o in result.get("intermediate_steps", []) if a.log.strip()]
if steps:
    print("\n" + "=" * 60)
    print(f"📥 История шагов агента ({len(steps)})")
    print("=" * 60)
    for i, (action, observation) in enumerate(steps, 1):
        obs_str = str(observation)
        preview = obs_str[:300] + ("..." if len(obs_str) > 300 else "")
        print(f"\n[{i}] {action.tool}")
        print(f"    Запрос:     {action.tool_input}")
        print(f"    Результат:  {preview}")
