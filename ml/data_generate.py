import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


n_samples = 1000

# Генерация признаков
distance_km = np.random.randint(10, 800, n_samples)           # расстояние от 10 до 800 км
traffic_index = np.random.randint(0, 11, n_samples)           # целое 0-10
driver_exp_months = np.random.randint(0, 240, n_samples)      # от 0 до 240 мес

# Логика формирования опоздания
# Чем больше расстояние и трафик, тем выше риск. Чем больше опыт, тем ниже риск.
risk_score = (
    0.03 * distance_km +                # расстояние влияет
    0.5 * traffic_index ** 2 +          # трафик квадратично
    -0.2 * driver_exp_months / 30 +     # опыт снижает риск
    np.random.normal(0, 5, n_samples)   # случайный шум
)
# Порог: если risk_score > 20, то опоздание (1)
delay = (risk_score > 20).astype(int)

# Собираем датасет
df = pd.DataFrame({
    'distance_km': distance_km,
    'traffic_index': traffic_index,
    'driver_exp_months': driver_exp_months,
    'delay': delay
})

# Сохраняем полный датасет
df.to_csv('data/delay_dataset.csv', index=False)
print(f"Сгенерировано {n_samples} записей.")

# Разделение на обучающую и тестовую выборки (80/20)
train_df, test_df = train_test_split(df, test_size=0.2)
train_df.to_csv('data/train_delay.csv', index=False)
test_df.to_csv('data/test_delay.csv', index=False)
