from llm_setup import build_llm
llm = build_llm()
response = llm.invoke("Say hello in one word.")
print(response.content)