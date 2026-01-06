from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

prompt = ChatPromptTemplate.from_template("""Crie um roteiro de viagem {numero_de_dias} dias,
para famialia com {numero_de_criancas} crianças, que gosta de {atividade}.""")

model = OllamaLLM(model="gpt-oss:20b-cloud", temperature=0.5)


chain = prompt | model

response  = chain.invoke({
    "numero_de_dias": numero_de_dias,
    "numero_de_criancas": numero_de_criancas,
    "atividade": atividade
})

print(response)

