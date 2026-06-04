import string
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from config import Config


def tokenize(s: str) -> list:
    return s.lower().translate(str.maketrans("", "", string.punctuation)).split()


def load_and_index_knowledge_base() -> BM25Retriever:
    """Загружает документы из KNOWLEDGE_DIR, разбивает на чанки и создаёт BM25Retriever"""
    documents = []
    knowledge_dir = Path(Config.KNOWLEDGE_DIR)

    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Папка с документами не найдена: {knowledge_dir}")

    for file_path in knowledge_dir.glob("*.*"):
        if file_path.suffix == '.txt':
            loader = TextLoader(str(file_path), encoding='utf-8')
        elif file_path.suffix == '.pdf':
            loader = PyPDFLoader(str(file_path))
        else:
            continue
        documents.extend(loader.load())

    if not documents:
        raise ValueError("В папке knowledge нет поддерживаемых документов (.txt, .pdf)")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)

    retriever = BM25Retriever.from_documents(
        documents=split_docs,
        preprocess_func=tokenize,
        k=3
    )
    return retriever


def get_rag_answer(query: str, retriever: BM25Retriever, llm) -> str:
    """
    Получает ответ на вопрос query, используя ретривер и LLM.
    Возвращает строку с ответом.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "В базе знаний не найдено информации по вашему запросу."

    context = "\n---\n".join([doc.page_content for doc in docs])
    from langchain_core.prompts import PromptTemplate
    prompt_template = PromptTemplate.from_template('''Ответь на вопрос пользователя. Используй только информацию из контекста. Если в контексте нет информации для ответа, сообщи об этом.
Контекст: {context}
Вопрос: {input}
Ответ:''')
    chain = prompt_template | llm
    answer = chain.invoke({"context": context, "input": query})
    return answer.content if hasattr(answer, 'content') else str(answer)