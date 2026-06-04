import streamlit as st
import re
import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="Labor Law KZ Assistant", layout="wide")
st.title("⚖️ ИИ-Ассистент по Трудовому кодексу РК")

@st.cache_resource
def load_system():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory="./db_knowledge", embedding_function=embeddings)
    # Оставляем llama3.1, но даем ей чуть больше свободы для подробного ответа
    llm = ChatOllama(model="llama3.1", temperature=0.3)
    return db, llm

db, llm = load_system()

def generate_expert_answer(query):
    # 1. СТРАТЕГИЯ ПОИСКА: Сначала ищем цифры (номер статьи)
    art_match = re.search(r'(\d+)', query)
    context_docs = []
    
    if art_match:
        art_num = art_match.group(1)
        # Прямой запрос к базе по ID статьи
        exact_results = db.get(where={"article_id": art_num})
        if exact_results and exact_results['documents']:
            context_docs = exact_results['documents']
            st.success(f"Найдена точная информация по Статье {art_num}")
    
    # 2. Если номер не помог или вопрос общий — ищем по смыслу
    if not context_docs:
        context_docs = [d.page_content for d in db.similarity_search(query, k=5)]

    context_text = "\n\n".join(context_docs)

    # 3. ПОДРОБНЫЙ СИСТЕМНЫЙ ПРОМПТ
    prompt = f"""
    ТЫ — ВЕДУЩИЙ ЮРИДИЧЕСКИЙ КОНСУЛЬТАНТ ПО ТРУДОВОМУ ПРАВУ КАЗАХСТАНА.
    ТВОЯ ЦЕЛЬ: Дать максимально подробный, развернутый и полезный ответ.

    ИСПОЛЬЗУЙ ТОЛЬКО ДАННЫЙ ТЕКСТ ИЗ ТК РК:
    {context_text}

    ОТВЕТЬ ПО ШАБЛОНУ:
    
    ### 📝 ПОДРОБНАЯ КОНСУЛЬТАЦИЯ
    - Развернуто объясни ситуацию своими словами.
    - Разбери права работника и обязанности работодателя.
    - Объясни все юридические тонкости простым языком.

    ### ⚖️ ПРАВОВОЕ ОСНОВАНИЕ (ИЗ ТК РК)
    - Укажи номер статьи.
    - Приведи ДОСЛОВНУЮ цитату из предоставленного текста.

    ### 🚀 РЕКОМЕНДАЦИЯ ДЛЯ ВАС
    - Дай пошаговый план: что делать пользователю (написать заявление, собрать акты, пойти в инспекцию и т.д.).

    ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}
    """
    
    return llm.invoke(prompt).content, context_text

# --- Интерфейс Чат-бота ---
if "history" not in st.session_state:
    st.session_state.history = []

if user_input := st.chat_input("Напишите ваш вопрос (например: Статья 80 или задержка зарплаты)"):
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Анализирую статьи закона и готовлю консультацию..."):
            answer, source = generate_expert_answer(user_input)
            st.markdown(answer)
            
            with st.expander("🔍 Посмотреть исходный текст из ТК РК"):
                st.write(source)
    
    st.session_state.history.append((user_input, answer))