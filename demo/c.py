
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
    llm = ChatOllama(model="llama3.1", temperature=0.3)
    return db, llm

db, llm = load_system()

# --- УЛУЧШЕННЫЙ ГЕНЕРАТОР КЛЮЧЕВЫХ СЛОВ ---
def get_legal_keywords(user_query):
    keyword_generator_prompt = f"""
    Преврати вопрос пользователя в 3-5 ключевых слов из Трудового кодекса РК для поиска. 
    Пример: 'начальник не платит' -> 'выплата заработной платы, сроки выплаты, статья 113'.
    Вопрос: {user_query}
    Ключевые слова:"""
    try:
        keywords = llm.invoke(keyword_generator_prompt).content
        return f"{user_query} {keywords}"
    except:
        return user_query

# --- ИСПРАВЛЕННАЯ ФУНКЦИЯ ГЕНЕРАЦИИ ОТВЕТА ---
def generate_expert_answer(query):
    # 1. СТРАТЕГИЯ ПОИСКА НОМЕРА СТАТЬИ (ИСПРАВЛЕНО)
    # Ищем число только если есть слова "статья", "ст" или если в запросе ТОЛЬКО число
    # Это исключит срабатывание на "2 года" или "10 вечера"
    art_num = None
    
    # Регулярка ищет: (слово статья/ст) + пробел? + число
    match = re.search(r'(?:статья|ст\.?|номер)\s*(\d+)', query.lower())
    if match:
        art_num = match.group(1)
    elif query.strip().isdigit(): # Если пользователь просто ввел "113"
        art_num = query.strip()

    context_docs = []
    
    # Если мы нашли РЕАЛЬНЫЙ номер статьи, ищем его жестко
    if art_num:
        exact_results = db.get(where={"article_id": art_num})
        if exact_results and exact_results['documents']:
            context_docs = exact_results['documents']
            # Если нашли, добавим еще пару похожих статей для контекста
            extra_docs = db.similarity_search(query, k=2)
            context_docs.extend([d.page_content for d in extra_docs])
    
    # 2. Если номера статьи нет (кейс), используем умный семантический поиск
    if not context_docs:
        refined_query = get_legal_keywords(query)
        # Увеличиваем k до 6, чтобы захватить больше смысла
        search_results = db.similarity_search(refined_query, k=6)
        context_docs = [d.page_content for d in search_results]

    context_text = "\n\n".join(context_docs)

    # 3. ТВОЙ СТАБИЛЬНЫЙ ПРОМПТ (БЕЗ ИЗМЕНЕНИЙ)
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