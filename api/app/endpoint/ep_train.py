# api/app/endpoint/ep_train.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
import os
from io import BytesIO

router = APIRouter(
    prefix="/api",
    tags=["training"],
)


@router.post("/train")
async def train_model(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    """
    Entraîne un modèle à partir d'un CSV préprocessé envoyé en upload,
    puis le sauvegarde sous models/<model_name>.joblib.
    """

    # 1) Vérifier le type de fichier
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un CSV.")

    # 2) Lire le contenu du fichier uploadé
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Impossible de lire le CSV : {e}")

    # 3) Vérification de la colonne cible
    target_col = "SiteEUIWN(kBtu/sf)"
    if target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Le CSV doit contenir la colonne cible : {target_col}"
        )

    # 4) Nettoyage éventuel d'un index sauvé par erreur
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 5) Séparation X / y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Assurer les types numériques pour les colonnes numériques attendues
    num_cols = [
        "DataYear",
        "NumberofBuildings",
        "NumberofFloors",
        "PropertyGFATotal",
        "PropertyGFAParking",
        "PropertyGFABuilding(s)",
        "LargestPropertyUseTypeGFA",
        "BuildingAge",
    ]
    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Vérifier les 3 colonnes catégorielles utilisées par le predict
    cat_cols = ["BuildingType", "PrimaryPropertyType", "Neighborhood"]
    missing = [c for c in (num_cols + cat_cols) if c not in X.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Colonnes manquantes dans le CSV pour l'entraînement: {missing}"
        )

    # Encodage One-Hot compatible avec ep_predict.format_input()
    X_encoded = pd.get_dummies(
        X,
        columns=cat_cols,
        prefix=["BuildingType", "PrimaryPropertyType", "Neighborhood"],
        prefix_sep="_",
        dummy_na=False
    )

    # Optionnel: remplacer NaN numériques par 0 (ou mieux: imputer selon votre stratégie)
    X_encoded = X_encoded.fillna(0)

    feature_names = X_encoded.columns.tolist()

    # 6) Entraîner un RandomForest sur X encodé
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_encoded, y)

    # 7) Sauvegarde
    os.makedirs("models", exist_ok=True)
    save_path = f"models/{model_name}.joblib"

    joblib.dump({"model": model, "features": feature_names}, save_path)