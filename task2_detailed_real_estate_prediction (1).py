
# Bài 2: Dự đoán giá bất động sản khu vực Hà Nội - Hà Đông

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, SimpleRNN, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 1. Tạo dataset với thông tin bất động sản cơ bản
def create_dataset():
    # Giả sử dữ liệu bao gồm vị trí (Hà Nội hoặc Hà Đông), diện tích (m2), và giá (ngàn VNĐ)
    data = {
        "location": ["Hanoi"] * 50 + ["Ha Dong"] * 50,
        "size": np.random.randint(30, 150, 100),  # diện tích ngẫu nhiên từ 30 đến 150 m2
        "price": np.random.randint(1_000, 10_000, 100)  # giá ngẫu nhiên từ 1 triệu đến 10 triệu VNĐ
    }
    df = pd.DataFrame(data)
    df["location"] = df["location"].apply(lambda x: 1 if x == "Hanoi" else 0)  # Biến đổi vị trí thành số
    return df

# 2. Tiền xử lý dữ liệu
def preprocess_data(df):
    # Chia dữ liệu thành đầu vào (X) và đầu ra (y)
    X = df[["location", "size"]].values
    y = df["price"].values.reshape(-1, 1)

    # Chuẩn hóa dữ liệu đầu vào và đầu ra
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Định hình lại dữ liệu cho các mô hình CNN, RNN, và LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, X_test, y_train, y_test, scaler_y

# 3. Xây dựng và huấn luyện các mô hình
def build_and_train_models(X_train, y_train):
    models = {
        "CNN": Sequential([
            Conv1D(32, 2, activation="relu", input_shape=(X_train.shape[1], 1)),
            Flatten(),
            Dense(1)
        ]),
        "RNN": Sequential([
            SimpleRNN(32, activation="relu", input_shape=(X_train.shape[1], 1)),
            Dense(1)
        ]),
        "LSTM": Sequential([
            LSTM(32, activation="relu", input_shape=(X_train.shape[1], 1)),
            Dense(1)
        ])
    }

    # Huấn luyện các mô hình
    for name, model in models.items():
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=10, verbose=1)
        print(f"Model {name} trained successfully!")

    return models

# 4. Đánh giá các mô hình
def evaluate_models(models, X_test, y_test, scaler_y):
    for name, model in models.items():
        predictions = model.predict(X_test)
        predictions = scaler_y.inverse_transform(predictions)  # Chuyển đổi lại giá trị dự đoán
        y_true = scaler_y.inverse_transform(y_test)  # Chuyển đổi lại giá trị thực

        mse = mean_squared_error(y_true, predictions)
        print(f"{name} Model Mean Squared Error: {mse}")

# Chạy các hàm trên
df = create_dataset()
X_train, X_test, y_train, y_test, scaler_y = preprocess_data(df)
models = build_and_train_models(X_train, y_train)
evaluate_models(models, X_test, y_test, scaler_y)
