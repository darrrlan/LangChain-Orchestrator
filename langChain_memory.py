from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.globals import set_debug

set_debug(True)

# Modelo
llm = OllamaLLM(
    model="gpt-oss:20b-cloud",
    temperature=0.5
)

# Prompt com histórico
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente de viagens especializado em turismo no Brasil."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Cadeia base (LCEL)
chain = prompt | llm

# Store de histórico (por sessão)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Runnable com memória
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Mensagens do usuário
mensagens = [
    "Quero visitar um lugar no Brasil famoso por suas praias e cultura. Pode me recomendar?",
    "Qual é o melhor período do ano para visitar em termos de clima?",
    "Quais tipos de atividades ao ar livre estão disponíveis?",
    "Alguma sugestão de acomodação eco-friendly por lá?",
    "Cite outras 20 cidades com características semelhantes às que descrevemos até agora.",
    "Na primeira cidade que você sugeriu lá atrás, quero saber 5 restaurantes para visitar."
]

# Execução
for msg in mensagens:
    resposta = conversation.invoke(
        {"input": msg},
        config={"configurable": {"session_id": "viagem-001"}}
    )
    print(f"Usuário: {msg}")
    print(f"AI: {resposta}")
    print("-" * 60)
