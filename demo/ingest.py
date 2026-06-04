import re
import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

def start_ingest():
    file_path = "tk_rk.txt"
    if not os.path.exists(file_path):
        print("❌ Файл tk_rk.txt не найден!")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Режем текст по статьям: ищем "Статья [номер]"
    article_parts = re.split(r'\n(?=Статья\s+\d+)', full_text)
    
    docs = []
    print(f"🔪 Разрезаю кодекс на статьи...")

    for part in article_parts:
        part = part.strip()
        if not part: continue
        
        # Извлекаем номер статьи
        number_match = re.search(r'Статья\s+(\d+)', part)
        art_num = number_match.group(1) if number_match else "unknown"

        # Создаем документ с привязкой к ID статьи
        doc = Document(
            page_content=part,
            metadata={"article_id": art_num}
        )
        docs.append(doc)

    # Очистка и создание базы
    if os.path.exists("./db_knowledge"):
        shutil.rmtree("./db_knowledge")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory="./db_knowledge"
    )
    print(f"🚀 ВСЕГО В БАЗЕ: {len(docs)} статей. Каждая статья теперь — уникальный объект.")

if __name__ == "__main__":
    start_ingest()