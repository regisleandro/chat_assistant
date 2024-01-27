from asyncio import sleep
import time
import streamlit as st
import openai
from dotenv import load_dotenv

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI

from pymongo import MongoClient

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

load_dotenv()

MONGO_CONNECTION = st.secrets['MONGO_CONNECTION']
DB_NAME = 'pdf_database'
COLLECTION_NAME = 'pdf_files'
INDEX_NAME = 'vector_search'
cluster = MongoClient(MONGO_CONNECTION)
MONGODB_COLLECTION = cluster[DB_NAME][COLLECTION_NAME]

def create_vector_search():
  embedding = HuggingFaceEmbeddings()
  vector_search = MongoDBAtlasVectorSearch.from_connection_string(
      MONGO_CONNECTION,
      f"{DB_NAME}.{COLLECTION_NAME}",
      embedding,
      index_name=INDEX_NAME
  )
  return vector_search

def perform_similarity_search(query, top_k=3):
  # Get the MongoDBAtlasVectorSearch object
  vector_search = create_vector_search()
  
  # Execute the similarity search with the given query
  documents = vector_search.similarity_search_with_score(
      query=query,
      k=top_k,
  )
  
  return documents

def create_system_message_template():
  template = """
    Você é um assistante online que irá responder perguntas utilizando somente este contexto: {context}.
    Se você não souber a resposta, ou não tiver algum contexto, apenas diga que não sabe, não tente inventar uma resposta.
    Procure responder a pergunta de forma clara e objetiva.
  """
  return SystemMessagePromptTemplate.from_template(template)

def answer_question(documents, question):
    system_message_prompt = create_system_message_template()
    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    model_name = 'gpt-3.5-turbo'
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    return llm(
        chat_prompt.format_prompt(
            context=documents, question=question
        ).to_messages()
    )


## Stremalit App
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title('💬💬💬 Hello bot')

if 'openai_model' not in st.session_state:
    st.session_state.openai_model = 'gpt-3.5-turbo'

if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'Olá, como posso ajudá-lo?'}]

prompt_template = """
    Você é um assistante online, voltado a agricultores, que irá responder: {}, utilizando como contexto: {}.
    Se você não souber a resposta, apenas diga que não sabe, não tente inventar uma resposta.
    Procure responder a pergunta de forma clara e objetiva.
  """

for message in st.session_state.messages:
  with st.chat_message(message['role']):
      st.markdown(message['content'])

if prompt := st.chat_input('Como posso ajudar?'):
  st.session_state.messages.append({'role': 'user', 'content': prompt})

  documents = perform_similarity_search(prompt, 5)
  list_documents = []
  for document in documents:
    list_documents.append(document[0].page_content)

  print(f"list_documents: {list_documents}")
  #st.session_state.messages.append({'role': 'user', 'content': prompt})

  # full_prompt = prompt_template.format(prompt, ", ".join(list_documents))
  with st.chat_message('user'):
    st.markdown(prompt)
  with st.chat_message('assistant'):
    with st.spinner('Estudando 📖 ...'):
      message_place_holder = st.empty()
      full_response = ''

      for chunk in answer_question(list_documents, prompt).content.split():
        full_response += chunk + " "
        time.sleep(0.1)
        message_place_holder.markdown(full_response + '▌')

      message_place_holder.markdown(full_response)

  st.session_state.messages.append({'role': 'assistant', 'content': full_response})
