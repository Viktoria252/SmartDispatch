import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


class DelayPredictor:
    def __init__(self):
        self.model = None

    def train(self, csv_path='data/train_delay.csv'):
        df = pd.read_csv(csv_path)
        X_train = df[['distance_km', 'traffic_index', 'driver_exp_months']]
        y_train = df['delay']
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, 'delay_model.pkl')

    def load(self):
        self.model = joblib.load('delay_model.pkl')

    def predict(self, distance_km, traffic_index, driver_exp_months):
        if self.model is None:
            if os.path.exists('delay_model.pkl'):
                self.load()
            else:
                # Если модели нет, пробуем обучить из CSV в папке data
                csv_path = 'data/train_delay.csv' if os.path.exists('data/train_delay.csv') else None
                self.train(csv_path)
        features = pd.DataFrame([[distance_km, traffic_index, driver_exp_months]],
                                columns=['distance_km', 'traffic_index', 'driver_exp_months'])
        prob = self.model.predict_proba(features)[0][1]
        return prob


def get_delay_risk(distance_km, traffic_index, driver_exp):
    predictor = DelayPredictor()
    return predictor.predict(distance_km, traffic_index, driver_exp)