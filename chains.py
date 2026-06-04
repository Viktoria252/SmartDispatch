from langchain_core.output_parsers import StrOutputParser
from config import Config
from langchain.prompts import PromptTemplate
from langchain_gigachat import GigaChat


llm = GigaChat(
    credentials=Config.GIGACHAT_CREDENTIALS,
    verify_ssl_certs=Config.GIGACHAT_VERIFY_SSL,
    model=Config.GIGACHAT_MODEL,
    async_mode=False
)

#Цепочка 1: парсинг заявки
parse_prompt = PromptTemplate(
    input_variables=["raw_text"],
    template="""
    Извлеки структурированные данные из заявки на перевозку груза.
    Ответ дай строго в формате JSON с ключами:
    origin, destination, weight_kg, cargo_type, deadline, special_requirements
    Текст: {raw_text}
    """
)
parse_chain = parse_prompt | llm | StrOutputParser()

# Цепочка 2: подбор машины (с использованием LLM)
vehicle_prompt = PromptTemplate(
    input_variables=["weight", "type", "deadline", "cars_db"],
    template="""
   Твоя задача: выбрать ОДНУ подходящую машину из списка ниже.
Условия: грузоподъёмность (capacity_kg) >= {weight} кг, статус 'available' (в таблице это поле status='available').
Из списка выбери лучший вариант (например, по минимальной ставке rate_km).

Данные автопарка (только доступные машины):
{cars_db}

Параметры заявки: вес = {weight} кг, тип груза = {type}, срок = {deadline}.

ВАЖНО: в ответе ОБЯЗАТЕЛЬНО укажи государственный номер (plate) выбранной машины.
Формат ответа строго такой:
"Машина НОМЕР (тип ТИП), ставка ЦЕНА руб/км, стоимость доставки = расстояние * ЦЕНА (расстояние пока не считаем)."
Пример: "Машина A777BB (тип Фургон), ставка 30 руб/км, стоимость доставки = 100 * 30 (расстояние пока не считаем)."

Ответ:
    """
)
vehicle_chain = vehicle_prompt | llm | StrOutputParser()