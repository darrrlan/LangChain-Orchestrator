from operator import itemgetter

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field


class Destino(BaseModel):
    cidade: str = Field(description="Cidade a visitar")
    motivo: str = Field(description="Motivo pelo qual é interessante visitar")


# Configuração do modelo LLM
model = OllamaLLM(
    model="gpt-oss:20b-cloud",
    temperature=0.5,
)

# Parser de saída estruturada
parser = JsonOutputParser(pydantic_object=Destino)

# Prompt para sugestão de cidade
prompt_cidade = PromptTemplate(
    template=(
        "Sugira uma cidade dado meu interesse por {interesse}.\n"
        "{formatacao_de_saida}"
    ),
    input_variables=["interesse"],
    partial_variables={
        "formatacao_de_saida": parser.get_format_instructions(),
    },
)

# Prompts auxiliares
prompt_restaurantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}."
)

prompt_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades culturais em {cidade}."
)

# Prompt final para agregação das informações
prompt_final = ChatPromptTemplate.from_messages(
    [
        ("ai", "Sugestão de viagem para a cidade: {cidade}"),
        ("ai", "Restaurantes que você não pode perder: {restaurantes}"),
        ("ai", "Atividades e locais culturais recomendados: {locais_culturais}"),
        (
            "system",
            "Combine as informações acima em dois parágrafos coerentes."
        ),
    ]
)

# Definição das chains
chain_cidade = prompt_cidade | model | parser
chain_restaurantes = prompt_restaurantes | model | StrOutputParser()
chain_cultural = prompt_cultural | model | StrOutputParser()
chain_final = prompt_final | model | StrOutputParser()

# Pipeline LCEL com execução paralela e agregação
chain = (
    chain_cidade
    | {
        "cidade": itemgetter("cidade"),
        "restaurantes": chain_restaurantes,
        "locais_culturais": chain_cultural,
    }
    | chain_final
)

# Execução
response = chain.invoke({"interesse": "praias"})
print(response)
