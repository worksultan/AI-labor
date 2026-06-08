import streamlit as st
import os
import re
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma

# 1. Настройка страницы
st.set_page_config(page_title="Labor Law KZ Assistant", layout="wide")
st.title("⚖️ ИИ-Ассистент по Трудовому кодексу РК")

@st.cache_resource
def load_system():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if not os.path.exists("./db_knowledge"):
        st.error("База данных не найдена! Сначала запустите ingest.py")
        return None, None
    db = Chroma(persist_directory="./db_knowledge", embedding_function=embeddings)
    # Оставляем 3.1, температура 0.2 (золотая середина)
    llm = ChatOllama(model="llama3.1", temperature=0.2)
    return db, llm

db, llm = load_system()

# --- УЛУЧШЕННЫЙ ГЕНЕРАТОР КЛЮЧЕВЫХ СЛОВ ---
def get_legal_keywords(user_query):
    keyword_generator_prompt = f"""
    Преврати вопрос пользователя в 3-5 ключевых слов из Трудового кодекса РК для поиска. 
    Пример: 'начальник не платит' -> 'выплата заработной платы, сроки выплаты, статья 113, пеня за задержку, статья 121'.
    Вопрос: {user_query}
    Ключевые слова:"""
    try:
        keywords = llm.invoke(keyword_generator_prompt).content
        return f"{user_query} {keywords}"
    except:
        return user_query

# --- ФУНКЦИЯ ГЕНЕРАЦИИ ОТВЕТА ---
def generate_expert_answer(query):
    # 1. СТРАТЕГИЯ ПОИСКА НОМЕРА СТАТЬИ
    art_num = None
    match = re.search(r'(?:статья|ст\.?|номер)\s*(\d+)', query.lower())
    if match:
        art_num = match.group(1)
    elif query.strip().isdigit():
        art_num = query.strip()

    context_docs = []
    
    # Поиск по точному ID
    if art_num:
        exact_results = db.get(where={"article_id": art_num})
        if exact_results and exact_results['documents']:
            context_docs = exact_results['documents']
            # К точной статье добавим еще 3 похожих для полноты картины
            extra_docs = db.similarity_search(query, k=3)
            context_docs.extend([d.page_content for d in extra_docs])
    
    # 2. Если кейс — ищем по смыслу
    if not context_docs:
        refined_query = get_legal_keywords(query)
        # Увеличиваем k до 8, чтобы найти минимум 2-3 разные статьи
        search_results = db.similarity_search(refined_query, k=8)
        context_docs = [d.page_content for d in search_results]

    context_text = "\n\n".join(context_docs)

    # 3. ТВОЙ СТАБИЛЬНЫЙ ПРОМПТ (с добавкой про несколько статей)
    prompt = f"""
    ТЫ — ВЕДУЩИЙ ЮРИДИЧЕСКИЙ КОНСУЛЬТАНТ ПО ТРУДОВОМУ ПРАВУ КАЗАХСТАНА.
    ТВОЯ ЦЕЛЬ: Дать максимально подробный, развернутый и полезный ответ.

    ИСПОЛЬЗУЙ ТОЛЬКО ДАННЫЙ ТЕКСТ ИЗ ТК РК:
    {context_text}

    ОТВЕТЬ ПО ШАБЛОНУ:
    
    ### 📝 ПОДРОБНАЯ КОНСУЛЬТАЦИЯ
    - Развернуто объясни ситуацию своими словами.
    - Разбери права работника и обязанности работодателя.
    - Если ситуация регулируется несколькими статьями, свяжи их логически.

    ### ⚖️ ПРАВОВОЕ ОСНОВАНИЕ (ИЗ ТК РК)
    - Укажи номера ВСЕХ подходящих статей (обычно 1-2 статьи).
    - Приведи ДОСЛОВНЫЕ цитаты для каждой найденной статьи.

    ### 🚀 РЕКОМЕНДАЦИЯ ДЛЯ ВАС
    - Дай пошаговый план: что делать пользователю.

    ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}
    """
    
    return llm.invoke(prompt).content, context_text

# --- ИНТЕРФЕЙС ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Напишите ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if db and llm:
            with st.spinner("Провожу глубокий юридический анализ..."):
                answer, source = generate_expert_answer(user_input)
                st.markdown(answer)
                with st.expander("🔍 Источники из ТК РК"):
                    st.write(source)
                st.session_state.messages.append({"role": "assistant", "content": answer})