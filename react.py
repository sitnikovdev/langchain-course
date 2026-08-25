from rich.console import Console
from rich.markdown import Markdown
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
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


# llm = ChatOllama(model="qwen3:8b", temperature=0)
console = Console()
# console.print(Markdown(result["output"]))


def get_text_length(text: str) -> int:
    """Return the lenght of a text by characters"""
    return len(text)


print("\n" + "=" * 60)
console.print(Markdown("👋 Hello ReAct LangChain!"))
console.print(get_text_length(text="Dog"))
print("=" * 60)
