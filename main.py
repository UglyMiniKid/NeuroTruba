import numpy as np
import pandas as pd
from neuro.model_creator import MLP
import random as rn
import threading
import time
from visual import start_server
from visual.server import set_data


print("Генерация синтетического датасета для прогноза неисправности трубы...")

np.random.seed(67)
n_samples = 10000

temp = np.random.uniform(-30, 80, n_samples)
pressure = np.random.uniform(1, 120, n_samples)
flow = np.random.uniform(50, 7000, n_samples)
vibration = np.random.uniform(0.05, 10.0, n_samples)

X = np.column_stack([temp, pressure, flow, vibration])

y = np.zeros(n_samples)

y[(temp > 55) & (pressure > 50)] = 1
y[(temp > 50) & (vibration > 4.5)] = 1
y[(temp > 65)] = 1
y[(temp < -20) & (pressure > 55)] = 1

y[(pressure > 85)] = 1
y[(pressure < 3)] = 1
y[(pressure > 60) & (vibration > 4.0)] = 1
y[(pressure > 55) & (flow > 5000)] = 1

y[(flow < 200)] = 1
y[(flow > 5500)] = 1
y[(flow > 4500) & (vibration > 4.5)] = 1
y[(flow > 4000) & (temp > 50) & (pressure > 50)] = 1

y[(vibration > 7.0)] = 1
y[(vibration > 5.0) & (temp > 45) & (pressure > 50)] = 1
y[(vibration > 5.0) & (flow > 4000)] = 1

y_2d = y.reshape(-1, 1)

print(f"Неисправностей: {y.sum()} из {n_samples} ({y.sum()/n_samples*100:.1f}%)")

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1
X_norm = (X - X_mean) / X_std

mlp = MLP(
    layers=[4, 256, 128, 64, 32, 1],
    activations=['relu', 'relu', 'relu', 'relu', 'sigmoid'],
    loss='binary_crossentropy',
    learning_rate=0.001,
    random_state=67
)

mlp.fit(X_norm, y_2d, epochs=1800, verbose=True)
print("Модель обучена.")


class Pipe:
    def __init__(self, pipe_id, temp, pressure, flow, vibration):
        self.pipe_id = pipe_id
        self.temp = temp
        self.pressure = pressure
        self.flow = flow
        self.vibration = vibration

    def update(self):
        if rn.random() < 0.05:
            self.temp += rn.uniform(-8, 8)
            self.pressure += rn.uniform(-10, 10)
            self.flow += rn.uniform(-800, 800)
            self.vibration += rn.uniform(-1.5, 1.5)
        else:
            self.temp += rn.uniform(-1.5, 1.5)
            self.pressure += rn.uniform(-2, 2)
            self.flow += rn.uniform(-150, 150)
            self.vibration += rn.uniform(-0.2, 0.2)

        self.temp = np.clip(self.temp, -9.9, 65.0)
        self.pressure = np.clip(self.pressure, 6.4, 99.9)
        self.flow = np.clip(self.flow, 130.1, 7999.0)
        self.vibration = np.clip(self.vibration, 0.2, 9.9)

    def get_array(self):
        return np.array([self.temp, self.pressure, self.flow, self.vibration])


pipes = [
    Pipe(1, 22.0, 32.0, 2500.0, 1.2),
    Pipe(2, 18.0, 28.0, 1800.0, 0.8),
    Pipe(3, 25.0, 35.0, 3200.0, 1.5),
    Pipe(4, 15.0, 25.0, 1500.0, 0.6),
    Pipe(5, 20.0, 30.0, 2000.0, 1.0),
]


def generate_prediction():
    while True:
        data = []
        for pipe in pipes:
            pipe.update()
            arr = pipe.get_array()
            arr_norm = (arr - X_mean) / X_std
            predict = mlp.predict(arr_norm.reshape(1, -1))
            predict_value = float(predict.item())

            data.append({
                "id": pipe.pipe_id,
                "temp": round(arr[0], 1),
                "pressure": round(arr[1], 1),
                "flow": round(arr[2], 0),
                "vibration": round(arr[3], 2),
                "status": "danger" if predict_value > 0.5 else "normal"
            })

        set_data(data)
        time.sleep(60)


threading.Thread(target=generate_prediction, daemon=True).start()

start_server()