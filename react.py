from typing import Union

from rich.console import Console
from rich.markdown import Markdown

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool, render_text_description
from langchain_ollama import ChatOllama
from langchain_classic.agents.output_parsers import (
    ReActSingleInputOutputParser,
)
from langchain_core.runnables import RunnablePassthrough


class CleanConsoleCallback(BaseCallbackHandler):
    def on_agent_action(self, action, **kwargs):
        print("\n" + "=" * 60)
        print(f"💭 {action.log.strip()}")
        print("=" * 60)

    def on_agent_finish(self, finish, **kwargs):
        print("\n" + "=" * 60)
        print("✅ FINAL ANSWER")
        print("=" * 60)


@tool
def get_text_length(text: str) -> int:
    """Return the length of a text by characters."""
    return len(text)


def format_log_to_str(
    intermediate_steps: list[tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    thoughts = ""

    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}" f"{observation}" f"\n{llm_prefix}"

    return thoughts


def find_tool_by_name(tools, tool_name: str):
    for current_tool in tools:
        if current_tool.name == tool_name:
            return current_tool

    raise ValueError(f"Tool with name {tool_name!r} not found")


console = Console()
tools = [get_text_length]

template = """
Answer the following question as best you can.

You have access to these tools:

{tools}

Use exactly this format.

Question: the input question
Thought: what you should think about
Action: one of [{tool_names}]
Action Input: the input for the tool
Observation: the result of the tool
Thought: what you think next
Final Answer: the final answer

Do not write Observation or Final Answer yourself after an Action.
After Action Input, stop generating.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template).partial(
    tools=render_text_description(tools),
    tool_names=", ".join(tool.name for tool in tools),
)

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    stop=["\nObservation:"],
)

agent = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x.get("intermediate_steps", []))
    )
    | prompt
    | llm
    | ReActSingleInputOutputParser()
)

question = "What is the length of DOG in characters?"
intermediate_steps = []

while True:
    agent_step: AgentAction | AgentFinish = agent.invoke(
        {
            "input": question,
            "intermediate_steps": intermediate_steps,
        }
    )

    if isinstance(agent_step, AgentFinish):
        console.print("FINAL ANSWER:")
        console.print(Markdown(agent_step.return_values["output"]))
        break

    if isinstance(agent_step, AgentAction):
        console.print(f"🔧 Tool: {agent_step.tool}")
        console.print(f"📥 Input: {agent_step.tool_input}")

        tool_to_use = find_tool_by_name(tools, agent_step.tool)
        observation = tool_to_use.invoke(agent_step.tool_input)

        console.print(f"📤 Observation: {observation}")

        intermediate_steps.append((agent_step, str(observation)))
