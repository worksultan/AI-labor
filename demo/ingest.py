import re
import os
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

def start_ingest():
    file_path = "tk_rk.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Режем текст строго по статьям (используем слово Статья как маркер)
    parts = re.split(r'\n(?=Статья\s+\d+)', text)
    
    docs = []
    for part in parts:
        part = part.strip()
        if not part: continue
        # Вытаскиваем номер статьи
        match = re.search(r'Статья\s+(\d+)', part)
        art_num = match.group(1) if match else ""
        
        # Сохраняем статью как отдельный документ с МЕТАДАННЫМИ
        doc = Document(
            page_content=part,
            metadata={"article_num": art_num}
        )
        docs.append(doc)

    # Создаем векторную базу (для общих вопросов)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists("./db_knowledge"):
        import shutil
        shutil.rmtree("./db_knowledge")
        
    Chroma.from_documents(docs, embeddings, persist_directory="./db_knowledge")
    print(f"✅ Готово! База создана. Всего статей в памяти: {len(docs)}")

if __name__ == "__main__":
    start_ingest()