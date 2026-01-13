from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser


class Destino(BaseModel):
    cidade: str = Field(description="Cidade a visitar")
    motivo: str = Field(description="Motivo pelo qual é interessante visitar")


model = OllamaLLM(
    model="gpt-oss:20b-cloud",
    temperature=0.5
)

parseador = JsonOutputParser(pydantic_object=Destino)


prompt_cidade = PromptTemplate(
    template="""
Sugira uma cidade dado meu interesse por {interesse}.
{formatacao_de_saida}
""",
    input_variables=["interesse"],
    partial_variables={
        "formatacao_de_saida": parseador.get_format_instructions()
    }
)

#chain = prompt_cidade | model | parseador
chain = prompt_cidade | model | parseador

response = chain.invoke({"interesse": "praias"})

print(response)
