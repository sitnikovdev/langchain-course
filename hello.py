from langchain_core.prompts.prompt import PromptTemplate
from langchain_ollama import ChatOllama

information = """
    In 2001, Musk became involved with the nonprofit Mars Society
    and discussed funding plans to place a growth-chamber for 
    plants on Mars.[90] Seeking a way to launch the greenhouse 
    payloads into space, Musk made two unsuccessful trips to Moscow
    to purchase intercontinental ballistic missiles (ICBMs) from Russian
    companies NPO Lavochkin and Kosmotras. Musk instead decided to 
    start a company to build affordable rockets.[91] With $100 million
    of his early fortune,[92] (equivalent to $180,000,000 in 2025) 
    Musk founded SpaceX in May 2002 and became the company's CEO and 
    Chief Engineer.
"""
summary_template = """
given the information {information} about a person I want to create:
1. A short summary
2. two interesting fact about them
"""

summary_promt_template = PromptTemplate(
    input_variables=["information"], template=summary_template
)

llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
)

chain = summary_promt_template | llm
response = chain.invoke(input={"information": information})
print(response.content)

# MARK: PERPHORMANCE BENCHMARK
metadata = response.response_metadata
prompt_tokens = response.usage_metadata["input_tokens"]
prompt_duration_ns = metadata["prompt_eval_duration"]
total_duration = metadata["total_duration"]
prompt_duration_seconds = total_duration / 1_000_000_000
prompt_tokens_per_second = prompt_tokens / prompt_duration_seconds

print(f"Время обработки prompt: {prompt_duration_seconds:.3f} сек.")
print(f"Скорость обработки: {prompt_tokens_per_second:.2f} токенов/сек.")
