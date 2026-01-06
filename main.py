
import os
from dotenv import load_dotenv
from ollama import Client

load_dotenv()
api_key = os.environ.get('OLLAMA_API_KEY')

client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + api_key}
)

numero_de_dias = 7
numero_de_criancas=2
atividade = "paria"
prompt = f"""Crie um roteiro de viagem {numero_de_dias} dias,
para famialia com {numero_de_criancas} crianças, que gosta de {atividade}."""


messages = [
  {
    'role': 'system', 'content':'teste',
    'content': prompt,
  },
]

for part in client.chat('gpt-oss:120b-cloud', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)
  

