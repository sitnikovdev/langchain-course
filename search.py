from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_tavily import TavilySearch

tools = [TavilySearch()]

llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Before calling any tool, always. "
            "Explain your reasoning in a short sentence. Use tools when needed."
            "When calling tavily_search, only use these exact values for time_range: "
            "'day', 'week', 'month', or 'year'. Never use any other value, and omit "
            "time_range entirely if unsure.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

result = agent_executor.invoke(
    {
        "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
    }
)

# for step in result["intermediate_steps"]:
# print(step)

print("Hello from search agent!")

# chain = summary_promt_template | llm
# response = chain.invoke(input={"information": information})
# print(response.content)

# MARK: PERPHORMANCE BENCHMARK
# metadata = response.response_metadata
# prompt_tokens = response.usage_metadata["input_tokens"]
# prompt_duration_ns = metadata["prompt_eval_duration"]
# total_duration = metadata["total_duration"]
# prompt_duration_seconds = total_duration / 1_000_000_000
# prompt_tokens_per_second = prompt_tokens / prompt_duration_seconds

# print(f"Время обработки prompt: {prompt_duration_seconds:.3f} сек.")
# print(f"Скорость обработки: {prompt_tokens_per_second:.2f} токенов/сек.")
