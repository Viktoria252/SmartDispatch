import requests
import io
import uuid
from pathlib import Path
from typing import Optional
from pydub import AudioSegment
import logging

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_salute_token(api_key: str) -> Optional[str]:
    # Создадим идентификатор UUID (36 знаков)
    rq_uid = str(uuid.uuid4())
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    # Заголовки
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'RqUID': rq_uid,
        'Authorization': f'Basic {api_key}'
    }

    # Тело запроса
    payload = {
        'scope': 'SALUTE_SPEECH_PERS'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, verify=False)
        if response != None:
            return response.json()['access_token']
    except requests.RequestException as e:
        logger.error(f"Ошибка: {e}")
        return None


def convert_to_pcm16(wav_path: Path, target_path: Path = None) -> Path:
    if target_path is None:
        target_path = wav_path.with_suffix(".raw")
    # Загружаем аудио
    audio = AudioSegment.from_file(wav_path)
    # Переводим в моно, 16 кГц
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    # Экспортируем как сырой PCM (без заголовка)
    audio.export(target_path, format="s16le")
    return target_path


def transcribe_audio_file(file_path: Path) -> Optional[str]:
    # 1. Получить токен
    token = get_salute_token(Config.SALUTE_SPEECH_API_KEY)
    if not token:
        print(token)
        logger.error("Не удалось получить токен SaluteSpeech")
        return None

    # 2. Привести аудио к нужному формату (PCM 16 кГц)
    try:
        pcm_file = convert_to_pcm16(file_path)
    except Exception as e:
        logger.error(f"Ошибка конвертации аудио: {e}")
        return None

    # 3. Отправить запрос на распознавание
    url = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "audio/x-pcm;bit=16;rate=16000"
    }

    with open(pcm_file, "rb") as f:
        audio_data = f.read()

    try:
        response = requests.post(url, headers=headers, data=audio_data, verify=False)
        if response.status_code == 200:
            result = response.json()
            # В ответе может быть поле "result" или "chunks" — проверяем
            if "result" in result:
                return result["result"]
            elif "chunks" in result:
                # иногда результат в виде списка фрагментов
                return " ".join([chunk.get("text", "") for chunk in result["chunks"]])
            else:
                logger.warning(f"Неожиданный формат ответа: {result}")
                return None
        else:
            logger.error(f"Ошибка API: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Исключение при запросе: {e}")
        return None
    finally:
        # удаляем временный PCM-файл
        if pcm_file.exists():
            pcm_file.unlink()

def synthesize_speech(text: str, token: str) -> bytes:
    """Отправляет текст в SaluteSpeech API и возвращает аудиоданные (MP3)."""
    # URL для синхронного синтеза речи
    url = "https://smartspeech.sber.ru/rest/v1/text:synthesize"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/text"
    }
    params = {
        "format": 'wav16',
        "voice": 'Nec_24000'
    }
    try:
        response = requests.post(url, headers=headers, params=params,  data=text,  verify=False)
        if response.status_code == 200:
            return response.content
        else:
            logger.warning(f"Неожиданный формат ответа: {response}")
            return None
    except Exception as e:
        logger.error(f"Исключение при запросе: {e}")
        return None

def text_to_speech_and_play(text: str, api_key: str):
    """
    Основная функция для озвучивания текста.
    Получает токен, синтезирует речь и проигрывает аудио.
    """
    # 1. Получаем access_token
    token = get_salute_token(api_key)
    if not token:
        logger.error("Не удалось авторизоваться в SaluteSpeech")
        return None

    # 2. Синтезируем речь из текста
    audio_data = synthesize_speech(text, token)
    if not audio_data:
        logger.error("Не удалось синтезировать речь")
        return None

    # 3. Преобразуем в байтовый поток для воспроизведения
    audio_bytes = io.BytesIO(audio_data)
    return audio_bytes