from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
    df_filtered = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2022 - 12 - 31")]

    # Convert date to matplotlib format
    df_filtered = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2022 - 12 - 31")].copy()
    df_filtered["date"] = df_filtered["date"].map(mdates.date2num)

    # *** Training and Prediction ***
    X = df[["open", "low", "high"]].values
    y = df["close"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=True
    )
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train_scaled, y_train)
    # Make predictions on the test set
    y_pred = linear_regressor.predict(X_test_scaled)
    # Evaluation
    mse = np.sqrt(np.mean((y_test - y_pred)**2))
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = r2_score(y_test, y_pred)
    print("Mean Square Error (MSE):", round(mse, 4))
    print("Mean Absolute Error (MAE):", round(mae, 4))
    print("R-squared:", round(r2, 4))

    # Lọc dữ liệu 2019 Q1
    df_2019_Q1 = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2019-03-31")].copy()
    # Chọn đúng các cột feature đã train
    X_2019 = df_2019_Q1[["open", "low", "high"]].values
    X_2019_scaled = scalar.transform(X_2019)
    y_2019 = df_2019_Q1["close"].values
    # Dự đoán bằng model sklearn
    y_pred_2019 = linear_regressor.predict(X_2019_scaled)
    # Gắn vào dataframe để tiện vẽ
    df_2019_Q1["predicted_close"] = y_pred_2019
    # Vẽ
    plt.figure(figsize=(12, 6))
    plt.plot(df_2019_Q1["date"], df_2019_Q1["close"], label="Actual Close Price", marker="o")
    plt.plot(df_2019_Q1["date"], df_2019_Q1["predicted_close"], label="Predicted Close Price(Model)",marker = "x")
    plt.title("Actual vs. Predicted Bitcoin Close Price (01/01 - 31/03/2019)")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()