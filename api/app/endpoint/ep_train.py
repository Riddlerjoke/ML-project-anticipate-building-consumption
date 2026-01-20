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

    feature_names = X.columns.tolist()

    # 6) Entraîner un RandomForest
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    # 7) Sauvegarde
    os.makedirs("models", exist_ok=True)
    save_path = f"models/{model_name}.joblib"

    joblib.dump({"model": model, "features": feature_names}, save_path)

    return {
        "message": "Modèle entraîné avec succès !",
        "model_saved_as": save_path,
        "nb_samples": len(df),
        "nb_features": len(feature_names)
    }
