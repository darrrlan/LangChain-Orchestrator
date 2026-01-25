from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


class Destino(BaseModel):
    cidade: str = Field(description="Cidade a visitar")
    motivo: str = Field(description="Motivo pelo qual é interessante visitar")


model = OllamaLLM(
    model="gpt-oss:20b-cloud",
    temperature=0.5
)

parseador = JsonOutputParser(pydantic_object=Destino)


prompt_cidade = PromptTemplate(
    template="""Sugira uma cidade dado meu interesse por {interesse}.
    {formatacao_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={
        "formatacao_de_saida": parseador.get_format_instructions()
    }
)

prompt_restaurantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}."
)

prompt_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades culturais em {cidade}."
)

chain1 = prompt_cidade | model | parseador
chain2 = prompt_restaurantes | model | StrOutputParser()
chain3 = prompt_cultural | model | StrOutputParser()

chain = (chain1 |{"restaurantes":chain2 , "locais_culturais": chain3}) # entre chaves que dizer que vao usar a saída da chain1 para chain 2 e 3



response = chain.invoke({"interesse": "praias"})

print(response)
