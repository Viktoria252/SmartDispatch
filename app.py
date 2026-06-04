import asyncio
import sys
import nest_asyncio

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

nest_asyncio.apply()

import streamlit as st
import tempfile
from pathlib import Path
import pandas as pd

from utils.transcription import transcribe_audio_file, text_to_speech_and_play
from chains import parse_chain
from agents import agent_executor, analyst_agent, memory
from rag.retriever import load_and_index_knowledge_base, get_rag_answer
from config import Config
from langchain_gigachat import GigaChat


# --- Функция для обработки сообщения пользователя (текст) ---
def process_user_message(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    memory.chat_memory.add_user_message(prompt)
    with st.spinner("Анализирую заявку..."):
        analysis = analyst_agent.run(prompt)
        st.info(f"Аналитик: {analysis}")

    with st.spinner("Ищу решение..."):
        # Агент обрабатывает запрос
        response = agent_executor.invoke({"input": prompt})
        assistant_reply = response['output']

        # Парсим данные для дополнительного отображения
        parsed = parse_chain.invoke({"raw_text": prompt})
        st.info(f"Извлечено из заявки: {parsed}")

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.write(assistant_reply)

    # Озвучивание ответа, если включено
    if st.session_state.get("use_tts", False):
        audio_bytes = text_to_speech_and_play(assistant_reply, Config.SALUTE_SPEECH_API_KEY)
        if audio_bytes:
            st.audio(audio_bytes, format='audio/mpeg')


@st.cache_resource
def get_cached_retriever():
    with st.spinner("Индексация базы знаний (первый раз может занять минуту)..."):
        return load_and_index_knowledge_base()

# --- Настройка страницы ---
st.set_page_config(page_title="SmartDispatch", layout="wide")
st.title("SmartDispatch: Виртуальный ассистент диспетчера")
# Инициализация ретривера
if "retriever" not in st.session_state:
    st.session_state.retriever = get_cached_retriever()
    st.success("База знаний готова!")

# Инициализация данных
if 'cars_df' not in st.session_state:
    st.session_state.cars_df = pd.read_csv(Config.CARS_DB_PATH)

# Боковая панель управления
with st.sidebar:
    st.header("Управление")
    uploaded_file = st.file_uploader("Загрузить аудио заявки", type=["wav", "mp3"])
    use_tts = st.checkbox("Озвучивать ответ", value=False, key="use_tts")
    st.subheader("Автопарк")
    st.dataframe(st.session_state.cars_df)

    # RAG система (отдельный запрос к базе знаний)
    st.subheader("База знаний (RAG)")
    rag_query = st.text_input("Спросить про правила перевозки")
    if rag_query:
        # Создаём LLM для ответа
        llm = GigaChat(
            credentials=Config.GIGACHAT_CREDENTIALS,
            verify_ssl_certs=Config.GIGACHAT_VERIFY_SSL,
            model=Config.GIGACHAT_MODEL,
            async_mode=False
        )
        answer = get_rag_answer(rag_query, st.session_state.retriever, llm)
        st.write(answer)

# Основной интерфейс чата
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Здравствуйте! Напишите или продиктуйте заявку на перевозку."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Обработка текстового ввода ---
if prompt := st.chat_input("Введите заявку"):
    process_user_message(prompt)

# --- Обработка аудио-файла ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = Path(tmp_file.name)

    with st.spinner("Распознаю речь..."):
        recognized_text = transcribe_audio_file(tmp_path)

    if recognized_text:
        # Добавляем в историю сообщение от пользователя
        st.session_state.messages.append({"role": "user", "content": f"🎤 {recognized_text}"})
        with st.chat_message("user"):
            st.write(f"Расшифровка аудио: {recognized_text}")

        # Обрабатываем распознанный текст как обычную заявку
        process_user_message(recognized_text)
    else:
        st.error("Не удалось распознать речь. Попробуйте другой аудиофайл.")

    # Удаляем временный файл
    tmp_path.unlink()
