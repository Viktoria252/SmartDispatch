from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.chains.llm import LLMChain

from chains import parse_chain, vehicle_chain
from ml.predictor import get_delay_risk
from config import Config
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_gigachat import GigaChat
import re
from langchain.memory import ConversationBufferWindowMemory


def parse_delay_input(x: str) -> str:
    """Извлекает три числа из строки и вызывает get_delay_risk"""
    # Находим все числа (целые и десятичные) в строке
    numbers = re.findall(r'\d+(?:\.\d+)?', x)
    if len(numbers) >= 3:
        try:
            distance = float(numbers[0])
            traffic = float(numbers[1])
            exp = float(numbers[2])
            risk = get_delay_risk(distance, traffic, exp)
            return f"Риск опоздания: {risk:.1%}"
        except Exception as e:
            return f"Ошибка при расчёте: {e}"
    else:
        return (f"Не удалось извлечь три числа из строки: '{x}'. "
                f"Укажите расстояние, трафик и опыт водителя (пример: '120, 5, 24')")

def select_vehicle(weight_type_deadline: str) -> str:
    """
        Ожидает строку: "вес, тип_груза, срок"
        Например: "1200, замороженные продукты, завтра"
        Если данные неполные, подставляются значения по умолчанию.
        """
    # Разбор входной строки
    parts = [p.strip() for p in weight_type_deadline.split(',')]
    weight = parts[0] if len(parts) > 0 else "1000"
    cargo_type = parts[1] if len(parts) > 1 else "стандартный"
    deadline = parts[2] if len(parts) > 2 else "сегодня"

    # Загружаем автопарк и превращаем в текст
    df = pd.read_csv(Config.CARS_DB_PATH)
    # Показываем только доступные машины для экономии токенов
    available_df = df[df['status'] == 'available']
    cars_db_str = available_df.to_string(index=False)

    # Вызываем цепочку
    try:
        result = vehicle_chain.invoke({
            "weight": weight,
            "type": cargo_type,
            "deadline": deadline,
            "cars_db": cars_db_str
        })
        return result.strip()
    except Exception as e:
        return f"Ошибка при подборе машины: {e}"


llm = GigaChat(
    credentials=Config.GIGACHAT_CREDENTIALS,
    verify_ssl_certs=Config.GIGACHAT_VERIFY_SSL,
    model=Config.GIGACHAT_MODEL,
    async_mode=False
)

# --- Добавление памяти ---
# Создаём память, которая хранит последние 5 взаимодействий.
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",  # Ключ для доступа к истории в промпте
    k=8,                        # Количество последних взаимодействий
    return_messages=True        # Возвращать в формате сообщений LangChain
)

# Инструменты для основного агента
tools = [
    Tool(name="ParseOrder",
         func=lambda x: parse_chain.invoke({"raw_text": x}),
         description="Парсит текст заявки в JSON с полями: origin, destination, weight_kg, cargo_type, deadline, special_requirements"),
    Tool(name="PredictDelay",
         func=parse_delay_input,
         description="Прогнозирует риск опоздания. Вход: расстояние (км), трафик (0-10), опыт водителя (мес). Пример: '120 км, трафик 5, опыт 24'"),
    Tool(name="SelectVehicle",
         func=select_vehicle,
         description="Подбирает машину. Вход: 'вес, тип_груза, срок' (через запятую). Пример: '1500, овощи, сегодня'")
]

#Стандартный ReAct промпт (содержит нужные переменные)
react_prompt = PromptTemplate.from_template("""Ты — агент-диспетчер грузоперевозок. Отвечай на вопросы пользователя, используя доступные инструменты.

Инструменты: {tools}

Чтобы выполнить задание, строго соблюдай следующий формат (заглавные буквы и двоеточия обязательны):

Question: вопрос пользователя
Thought: твои рассуждения на русском языке, что нужно сделать и какой инструмент применить
Action: название инструмента из [{tool_names}]
Action Input: входные данные для инструмента (строка)
Observation: результат от инструмента
... (повторяй Thought/Action/Action Input/Observation столько раз, сколько нужно)
Thought: теперь я знаю ответ
Final Answer: итоговый ответ пользователю на русском языке

Важно! Все поля (Thought, Final Answer и т.д.) пиши на русском языке.

История разговора:
{chat_history}

Вопрос пользователя: {input}
{agent_scratchpad}
""")

# Основной агент (исполнитель)
main_agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(
    agent=main_agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# Второй агент (аналитик) – просто LLMChain без инструментов
analyst_prompt = PromptTemplate.from_template("""
Ты аналитик. Проверь заявку: {input}. 
История разговора:
{chat_history}
Если не хватает данных (например, вес, маршрут, тип груза), задай уточняющий вопрос.
Не будь очень придирчивым, нам нужны только следующие данные: вес, тип груза, маршрут, сроки и дополнительная информация (необязательно).
Если данных достаточно, ответь: "Заявка полная" и перечисли параметры.
""")
analyst_chain = LLMChain(
    llm=llm,
    prompt=analyst_prompt,
    memory=memory,
    verbose=False
)

class SimpleAnalystAgent:
    """Обёртка для единообразного вызова"""
    def run(self, user_input):
        result = analyst_chain.run(input=user_input)
        memory.chat_memory.add_ai_message(f"[Аналитик]: {result}")
        return result

analyst_agent = SimpleAnalystAgent()