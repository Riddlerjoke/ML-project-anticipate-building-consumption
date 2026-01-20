# api/app/train_model.py
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

CSV_PATH = "data/dataset_model_ready.csv"
TARGET_COL = "SiteEUIWN(kBtu/sf)"
DEFAULT_MODEL_PATH = "models/energy_model.joblib"


def train_and_save(
    csv_path: str = CSV_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
):
    # 1) Charger le CSV
    df = pd.read_csv(csv_path)

    # 2) Nettoyer l’index perdu
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 3) Vérifier la cible
    if TARGET_COL not in df.columns:
        raise ValueError(f"La colonne cible {TARGET_COL} est manquante dans {csv_path}")

    # 4) X / y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    feature_names = X.columns.tolist()

    # 5) Split pour évaluer rapidement le modèle
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6) Entraîner le RandomForest (même config que l’endpoint)
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 7) Petite éval pour info
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"[train_model.py] RMSE test : {rmse:.2f} | R² : {r2:.3f}")

    # 8) Sauvegarde
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": model, "features": feature_names}, model_path)

    print(f"Modèle entraîné et sauvegardé dans : {model_path}")


if __name__ == "__main__":
    train_and_save()
