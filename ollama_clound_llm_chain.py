from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model="gpt-oss:20b-cloud", temperature=0.5)

prompt_cidade = ChatPromptTemplate.from_template(
    "Sugira uma cidade dado meu interesse por {interesse}. A sua saída deve ser SOMENTE o nome da cidade. Cidade:"
)

prompt_restaurantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}."
)

prompt_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades culturais em {cidade}."
)


chain = prompt_cidade | model | prompt_restaurantes | model | prompt_cultural | model 


response = chain.invoke({"interesse": "praias"})

print(response)
