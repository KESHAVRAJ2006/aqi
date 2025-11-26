import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(path="data/aqi_sample.csv"):
    return pd.read_csv(path)

def preprocess(df):
    df = df.ffill().fillna(0)
    if "date" in df.columns: df = df.drop(columns=["date"])
    return df

def main():
    df = load_data()
    df = preprocess(df)
    X = df.drop("AQI", axis=1)
    y = df["AQI"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    print("Validation R²:", model.score(X_val, y_val))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/aqi_model.pkl")

if __name__ == "__main__":
    main()
