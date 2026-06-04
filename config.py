import os
from pathlib import Path
import pandas as pd

class Config:
    # GigaChat
    GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "MDE5ZTI1OTctZWQ1OC03M2QxLWJkZWItOTJjMWJlMTk2Mjk2OjgyZTQ0ZGQyLTljZWItNDhkMS04Y2U1LTRmYjAzY2I3ZWNiYQ==")
    GIGACHAT_VERIFY_SSL = False

    # SaluteSpeech
    SALUTE_SPEECH_API_KEY = os.getenv("SALUTE_SPEECH_API_KEY", "MDE5ZTI5ZjctNjY0YS03OGVjLWFhNmYtYTU1MTFkOTY3M2JjOjNmNjUyZmJmLWI5YTgtNDIyNy04NDRkLTZhMGM4MjljZWIzNg==")

    #HuggingFace
    HUGGING_FACE_API_KEY = os.getenv("HF_TOKEN", "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")

    BASE_DIR = Path(__file__).parent
    CARS_DB_PATH = BASE_DIR / "data" / "cars.csv"
    KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"

    GIGACHAT_MODEL = "GigaChat-2"
    EMBEDDINGS_MODEL = "Embeddings"