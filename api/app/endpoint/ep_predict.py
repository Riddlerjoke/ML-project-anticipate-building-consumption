# api/app/endpoint/ep_predict.py
from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
import os

from api.pydantic.schemas import BuildingFeatures

router = APIRouter(
    prefix="/api",
    tags=["prediction"],
)

# Modèle et features actuellement chargés en mémoire pour la prédiction
CURRENT_MODEL = None
CURRENT_FEATURES = None


def _extract_allowed(prefix: str) -> set[str]:
    global CURRENT_FEATURES
    if CURRENT_FEATURES is None:
        return set()
    prefix_ = prefix + "_"
    return {f[len(prefix_):] for f in CURRENT_FEATURES if f.startswith(prefix_)}


def _require_in_allowed(value: str, allowed: set[str], field_label: str):
    if value not in allowed:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Valeur catégorielle inconnue",
                "field": field_label,
                "value": value,
                "allowed_values": sorted(allowed),
            }
        )


def format_input(payload: BuildingFeatures) -> pd.DataFrame:
    global CURRENT_FEATURES

    if CURRENT_FEATURES is None:
        raise HTTPException(
            status_code=400,
            detail="Aucun modèle n'est chargé : les features ne sont pas définies. Appelez d'abord /api/load_model."
        )

    # Initialiser toutes les features à 0
    row = {feat: 0.0 for feat in CURRENT_FEATURES}

    # Numériques
    row["DataYear"] = payload.DataYear
    row["NumberofBuildings"] = float(payload.NumberofBuildings)
    row["NumberofFloors"] = float(payload.NumberofFloors)
    row["PropertyGFATotal"] = float(payload.PropertyGFATotal)
    row["PropertyGFAParking"] = float(payload.PropertyGFAParking)
    row["PropertyGFABuilding(s)"] = float(payload.PropertyGFABuilding_s)
    row["LargestPropertyUseTypeGFA"] = float(payload.LargestPropertyUseTypeGFA)
    row["BuildingAge"] = float(payload.BuildingAge)

    # Validation des catégorielles à partir des OHE présentes
    allowed_bt = _extract_allowed("BuildingType")
    allowed_ppt = _extract_allowed("PrimaryPropertyType")
    allowed_n  = _extract_allowed("Neighborhood")

    _require_in_allowed(payload.BuildingType, allowed_bt, "BuildingType")
    _require_in_allowed(payload.PrimaryPropertyType, allowed_ppt, "PrimaryPropertyType")
    _require_in_allowed(payload.Neighborhood, allowed_n, "Neighborhood")

    # Activation One-Hot (garantie existante)
    bt_col = f"BuildingType_{payload.BuildingType}"
    ppt_col = f"PrimaryPropertyType_{payload.PrimaryPropertyType}"
    n_col = f"Neighborhood_{payload.Neighborhood}"

    row[bt_col] = 1.0
    row[ppt_col] = 1.0
    row[n_col] = 1.0

    # Construire X dans l'ordre exact attendu
    X_row = pd.DataFrame([[row[col] for col in CURRENT_FEATURES]], columns=CURRENT_FEATURES)
    return X_row


@router.post("/predict")
def predict_energy(payload: BuildingFeatures):
    """
    Endpoint d'inférence : prédit la consommation énergétique à partir d'un bâtiment.
    Nécessite qu'un modèle ait été chargé auparavant via /api/load_model.
    """
    global CURRENT_MODEL

    if CURRENT_MODEL is None:
        raise HTTPException(
            status_code=400,
            detail="Aucun modèle n'est chargé ! Utilisez d'abord /api/load_model."
        )

    try:
        X_row = format_input(payload)
        prediction = float(CURRENT_MODEL.predict(X_row)[0])

        return {
            "prediction": prediction,
            "unit": "kBtu/sf/year",
            "unit_label": "EUI du site ajustée aux conditions météo"
        }
    except HTTPException:
        # On laisse remonter les HTTPException telles quelles
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")


@router.post("/load_model")
def load_model(model_name: str):
    """
    Charge en mémoire un modèle existant (pré-entraîné et sauvegardé),
    qui sera ensuite utilisé par /predict.
    """
    global CURRENT_MODEL, CURRENT_FEATURES

    path = f"models/{model_name}.joblib"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Modèle introuvable : {path}")

    try:
        artifact = joblib.load(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Impossible de charger le modèle : {e}")

    CURRENT_MODEL = artifact["model"]
    CURRENT_FEATURES = artifact["features"]

    return {
        "message": f"Modèle {model_name} chargé avec succès.",
        "nb_features": len(CURRENT_FEATURES)
    }


@router.get("/allowed_categories")
def allowed_categories():
    global CURRENT_FEATURES
    if CURRENT_FEATURES is None:
        raise HTTPException(status_code=400, detail="Aucun modèle n'est chargé !")

    def extract(prefix: str):
        return sorted(_extract_allowed(prefix))

    return {
        "BuildingType": extract("BuildingType"),
        "PrimaryPropertyType": extract("PrimaryPropertyType"),
        "Neighborhood": extract("Neighborhood"),
    }