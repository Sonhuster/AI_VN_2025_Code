import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def predict(X, w, b):
    return X.dot(w) + b


def gradient(y_hat, y, x):
    n = len(y)
    loss = y_hat - y
    dw = (x.T.dot(loss)) / n
    db = np.sum(loss) / n
    cost = np.sum(loss ** 2) / (2 * n)
    return dw, db, cost


def update_weight(w, b, lr, dw, db):
    w_new = w - lr * dw
    b_new = b - lr * db
    return w_new, b_new


def linear_regression_vectorized(X, y, learning_rate=0.01, num_iterations=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    losses = []

    for i in range(num_iterations):
        y_hat = predict(X, w, b)
        dw, db, cost = gradient(y_hat, y, X)
        w, b = update_weight(w, b, learning_rate, dw, db)
        losses.append(cost)

        if i % 50 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}")

    return w, b, losses
if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\User\Desktop\Akselos_Project\my\AI_VN\21_Deeplearning\BTC-Daily.csv")
    df = df.drop_duplicates()
    df.head()

    df["date"] = pd.to_datetime(df["date"])
    date_range = str(df["date"].dt.date.min()) + " to " + str(df["date"].dt.date.max())
    print(date_range)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Filter data for 2019-2022
    df_filtered = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2022-12-31")]

    # Convert date to matplotlib format
    df_filtered = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2022-12-31")].copy()
    df_filtered["date"] = df_filtered["date"].map(mdates.date2num)

    # Create the candlestick chart
    fig, ax = plt.subplots(figsize=(20, 6))

    candlestick_ohlc(ax, df_filtered[["date", "open", "high", "low", "close"]].values, width=0.6, colorup="g",
                     colordown="r")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    plt.title("Bitcoin Candlestick Chart (2019-2022)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.close()

    # Filter data for 2019-2022
    df_filtered = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2022-12-31")]

    # Convert date to matplotlib format
    df_filtered = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2022-12-31")].copy()
    df_filtered["date"] = df_filtered["date"].map(mdates.date2num)

    # *** Training and Prediction ***
    X = df_filtered[["open", "low", "high"]].values
    y = df_filtered["close"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=True
    )
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    w, b, losses = linear_regression_vectorized(X_train_scaled, y_train, learning_rate=0.01,
                                                num_iterations=5000)
    # Plot the loss function
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Function during Gradient Descent")
    plt.show()

    y_pred = predict(X_test, w, b)
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    # Calculate MAE
    mae = np.mean(np.abs(y_pred - y_test))
    # Calculate MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Calculate R-squared on training data
    y_train_pred = predict(X_train, w, b)
    train_accuracy = r2_score(y_train, y_train_pred)

    # Calculate R-squared on testing data
    test_accuracy = r2_score(y_test, y_pred)

    print("Root Mean Square Error (RMSE):", round(rmse, 4))
    print("Mean Absolute Error (MAE):", round(mae, 4))
    print("Training Accuracy (R-squared):", round(train_accuracy, 4))
    print("Testing Accuracy (R-squared):", round(test_accuracy, 4))

    # Lọc dữ liệu 2019 Q1
    df_2019_Q1 = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2019-03-31")].copy()

    # Chọn đúng các cột feature đã train
    X_2019 = df_2019_Q1[["open", "low", "high"]].values
    X_2019_scaled = scalar.transform(X_2019)  # dùng scalar đã fit trước đó
    y_2019 = df_2019_Q1["close"].values
    # Dự đoán bằng mô hình
    y_pred_2019 = predict(X_2019_scaled, w, b)
    # Gắn vào dataframe để dễ vẽ
    df_2019_Q1["predicted_close"] = y_pred_2019
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    plt.plot(df_2019_Q1["date"], df_2019_Q1["close"], label="Actual Close Price", marker="o")
    plt.plot(df_2019_Q1["date"], df_2019_Q1["predicted_close"], label="Predicted Close Price(Model)", marker = "x")
    plt.title("Actual vs. Predicted Bitcoin Close Price (01/01 - 31/03/2019)")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()