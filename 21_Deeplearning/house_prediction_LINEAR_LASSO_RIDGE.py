import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    #importing Dataset
    house_df = pd.read_csv(r"C:\Users\User\Desktop\Akselos_Project\my\AI_VN\21_Deeplearning\train-house-prices-advanced-regression-techniques.csv")
    #checking dataset
    house_df.head()

    house_df.shape
    house_df.describe()
    # # target variable
    # sns.distplot(house_df["SalePrice"])
    # plt.axvline(x=house_df["SalePrice"].mean(), linestyle="--",linewidth=2)
    # plt.title("Sales")
    # plt.show()

    # Checking missing values (horizontal view)
    missing = house_df.isnull().sum()
    missing = missing[missing > 0]
    missing = missing.sort_values(ascending=False)
    # plt.figure(figsize=(10, 8))
    # missing.plot.barh(color="skyblue", edgecolor="black")
    # plt.title("Missing Data by Feature", fontsize=14)
    # plt.xlabel("Number of Missing Values")
    # plt.ylabel("Feature Name")
    # plt.grid(axis="x", linestyle="--", alpha=0.7)
    # plt.gca().invert_yaxis()
    # plt.show()

    # plt.figure(figsize=(30, 9))
    # sns.heatmap(house_df.corr(numeric_only=True),cmap="coolwarm",linewidths=0.5,center=0,cbar_kws={"shrink": 0.8})
    # plt.title("Correlation Heatmap of Numerical Features", fontsize=16, pad=15)
    # plt.show()

    # Chọn 8 đặc trưng numeric quan trọng nhất dựa trên tương quan với SalePrice
    important_cols = [
    "OverallQual", "GrLivArea", "GarageCars", "GarageArea",
    "TotalBsmtSF", "1stFlrSF", "FullBath", "YearBuilt"
    ]
    # fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    # axes = axes.flatten()
    # for i, col in enumerate(important_cols):
    #     sns.boxplot(data=house_df, y=col, ax=axes[i], color="skyblue")
    #     axes[i].set_title(col, fontsize=12, fontweight="bold")
    #     axes[i].tick_params(labelsize=9)
    #
    # for i in range(len(important_cols), len(axes)):
    #     axes[i].set_visible(False)
    #
    # plt.suptitle("Boxplots of Top 8 Numerical Features", fontsize=16,
    # fontweight="bold", y=1.03)
    # plt.tight_layout()
    # plt.show()
    house_df = house_df.drop(["Id","Alley","PoolQC","Fence","MiscFeature"], axis=1)

    # create training and validation sets
    train_df, test_df = train_test_split(
        house_df,
        test_size=0.25,
        random_state=42
    )

    y_train = train_df["SalePrice"].values
    y_test = test_df["SalePrice"].values
    train_df = train_df.drop(["SalePrice"], axis=1)
    test_df = test_df.drop(["SalePrice"], axis=1)

    num_cols = [col for col in train_df.columns if train_df[col].dtype in ["float64", "int64"]]
    cat_cols = [col for col in train_df.columns if train_df[col].dtype not in ["float64", "int64"]]

    # fill none for categorical columns
    train_df[cat_cols] = train_df[cat_cols].fillna("none")
    test_df[cat_cols] = test_df[cat_cols].fillna("none")

    # One-hot encode categorical columns
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[cat_cols])

    encoded_cols = list(encoder.get_feature_names_out(cat_cols))
    train_df[encoded_cols] = encoder.transform(train_df[cat_cols])
    test_df[encoded_cols] = encoder.transform(test_df[cat_cols])

    imputer = SimpleImputer()
    train_df[num_cols] = imputer.fit_transform(train_df[num_cols])
    test_df[num_cols] = imputer.transform(test_df[num_cols])

    # scaler = MinMaxScaler()
    # train_num_features = scaler.fit_transform(train_df[num_cols])
    # test_num_features = scaler.transform(test_df[num_cols])
    #
    # X_train = np.hstack([train_num_features, train_df[encoded_cols].values])
    # X_test = np.hstack([test_num_features, test_df[encoded_cols].values])
    #
    # models = {
    #     'LinearRegression': LinearRegression(),
    #     'Ridge': Ridge(),
    #     'Lasso': Lasso()
    # }
    #
    # # Khởi tạo list lưu kết quả
    # train_rmse_results = []
    # test_rmse_results = []
    # train_r2_results = []
    # test_r2_results = []
    # model_names = []
    #
    # # Huấn luyện và tính metric
    # for name, model in models.items():
    #     regressor = model
    #     regressor.fit(X_train, y_train.reshape(-1, 1))
    #     # Dự đoán
    #     y_train_pred = regressor.predict(X_train)
    #     y_test_pred = regressor.predict(X_test)
    #     # Tính RMSE
    #     train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    #     test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    #     # Tính r2
    #     train_r2 = r2_score(y_train, y_train_pred)
    #     test_r2 = r2_score(y_test, y_test_pred)
    #
    #     # Lưu kết quả
    #     model_names.append(name)
    #     train_rmse_results.append(train_rmse)
    #     test_rmse_results.append(test_rmse)
    #     train_r2_results.append(train_r2)
    #     test_r2_results.append(test_r2)
    #
    # # Tạo DataFrame tổng hợp
    # df_results = pd.DataFrame({
    #     "Model": model_names,
    #     "Train_RMSE": train_rmse_results,
    #     "Test_RMSE": test_rmse_results,
    #     "Train_R2": train_r2_results,
    #     "Test_R2": test_r2_results
    # }).sort_values(by="Test_R2", ascending=False)
    #
    # print(df_results)

    poly_features = PolynomialFeatures(
        degree=2, interaction_only=True, include_bias=False
    )

    train_poly_features = poly_features.fit_transform(train_df[num_cols])
    test_poly_features = poly_features.transform(test_df[num_cols])

    scaler = MinMaxScaler()
    train_poly_features = scaler.fit_transform(train_poly_features)
    test_poly_features = scaler.transform(test_poly_features)

    X_train_poly = np.hstack([train_poly_features, train_df[encoded_cols].values])
    X_test_poly = np.hstack([test_poly_features, test_df[encoded_cols].values])

    # making dictionary of models
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'LinearRegression': LinearRegression()
    }

    # lists to store results
    r2_results = []
    rmse_results = []
    train_rmse_results = []
    train_r2_results = []
    model_names = []
    # training and evaluating each model
    for name, model in models.items():
        regressor = model
        regressor.fit(X_train_poly, y_train.reshape(-1, 1))

        y_train_pred = regressor.predict(X_train_poly)
        y_test_pred = regressor.predict(X_test_poly)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # store
        model_names.append(name)
        train_rmse_results.append(train_rmse)
        rmse_results.append(test_rmse)
        train_r2_results.append(train_r2)
        r2_results.append(test_r2)

    # create dataframe
    df_results = pd.DataFrame({
        "Model": model_names,
        "Train_RMSE": train_rmse_results,
        "Test_RMSE": rmse_results,
        "Train_R2": train_r2_results,
        "Test_R2": r2_results
    }).sort_values(by="Test_R2", ascending=False)
    print(df_results)