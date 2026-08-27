from rich.console import Console
from rich.markdown import Markdown
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_tavily import TavilySearch
from langchain_core.callbacks import BaseCallbackHandler


class CleanConsoleCallback(BaseCallbackHandler):
    """Печатает только осмысленные шаги: что запросил агент и что реально пришло."""

    def on_agent_action(self, action, **kwargs):
        if not action.log.strip():
            return
        print(f"\n{'─'*60}")
        print(f"🔧 Вызов инструмента: {action.tool}")
        print(f"   Аргументы: {action.tool_input}")

    def on_tool_end(self, output, **kwargs):
        # Реальный результат инструмента, а не пустой лог действия
        preview = str(output)[:400]
        print(f"📥 Результат: {preview}{'...' if len(str(output)) > 400 else ''}")

    def on_agent_finish(self, finish, **kwargs):
        print(f"\n{'─'*60}")
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
    # Оставляем только релевантные поля вместо сырого дампа объекта
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

print("\n" + "=" * 60)
print("📥 История шагов агента")
print("=" * 60)
for i, (action, observation) in enumerate(
    [(a, o) for a, o in result.get("intermediate_steps", []) if a.log.strip()], 1
):
    print(f"\n[{i}] {action.tool} → {action.tool_input}")
    print(f"    {str(observation)[:300]}")