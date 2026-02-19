# main.py
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from api.app.endpoint import ep_predict, ep_train


app = FastAPI(
    title="Energy Prediction API",
    version="1.0.0",
    description=(
        "Consultez le notebook : "
        "[Notebook EDA/Modélisation](/static/notebook_preprocessing_modelisation_final.html)\n\n"
        "Ou via la route : [/docs/notebook](/docs/notebook)"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# On monte les routes
app.include_router(ep_train.router, prefix="/train", tags=["training"])
app.include_router(ep_predict.router, prefix="/predict", tags=["prediction"])


@app.get("/")
def root():
    return {"message": "API OK "}


# Sert /static/... depuis ./notebook
app.mount("/static", StaticFiles(directory="notebook"), name="static")


# Route directe vers le HTML exporté
@app.get("/docs/notebook", include_in_schema=False)
def notebook_page():
    html_path = Path("notebook/notebook_preprocessing_modelisation_final.html")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))