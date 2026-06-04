import streamlit as st
import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Настройка страницы
st.set_page_config(page_title="Labor Law KZ Expert Pro", layout="wide")
st.title("⚖️ ИИ-Эксперт ТК РК (Версия Llama 3.1 + Hybrid Search)")

@st.cache_resource
def load_system():
    # Эмбеддинги
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Загружаем векторную базу
    if not os.path.exists("./db_knowledge"):
        st.error("База знаний не найдена! Запусти ingest.py")
        return None
    
    vectorstore = Chroma(persist_directory="./db_knowledge", embedding_function=embeddings)
    
    # --- НАСТРОЙКА ГИБРИДНОГО ПОИСКА ---
    all_docs_data = vectorstore.get()
    all_texts = all_docs_data["documents"]
    
    from langchain_core.documents import Document
    doc_objects = [Document(page_content=text) for text in all_texts]
    
    # Ключевой поиск (BM25) - находит точные слова "зарплата", "статья 113"
    keyword_retriever = BM25Retriever.from_documents(doc_objects)
    keyword_retriever.k = 5
    
    # Смысловой поиск (Векторный) - находит общие темы
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Объединяем: 70% веса даем ключевым словам (для законов это важнее!)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[keyword_retriever, vector_retriever],
        weights=[0.7, 0.3]
    )
    
    # --- НАСТРОЙКА МОДЕЛИ LLAMA 3.1 ---
    llm = ChatOllama(model="llama3.1", temperature=0)
    
    system_prompt = (
        "Ты — высококвалифицированный юрист Республики Казахстан. Твоя база — Трудовой кодекс РК.\n"
        "ИНСТРУКЦИЯ:\n"
        "1. Используй ПРЕДОСТАВЛЕННЫЙ ТЕКСТ ниже для ответа.\n"
        "2. Если в тексте есть конкретные статьи (например, ст. 113 или 121), цитируй их.\n"
        "3. Если информации в тексте НЕТ, ответь: 'К сожалению, в моей базе ТК РК нет информации по этому вопросу'.\n"
        "4. Отвечай официально и по делу. Не используй внешние знания.\n\n"
        "СТАТЬИ ТК РК ДЛЯ АНАЛИЗА:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(ensemble_retriever, chain)

qa_system = load_system()

# --- Интерфейс чата ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Напишите ваш вопрос (например: Статья 113 или задержка зарплаты)"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Юридический анализ документации..."):
            response = qa_system.invoke({"input": user_input})
            st.markdown(response["answer"])
            
            with st.expander("📚 Использованные фрагменты ТК РК:"):
                for doc in response["context"]:
                    st.write(doc.page_content)
                    st.divider()
    
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})