# main.py
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.app.endpoint import ep_predict, ep_train


app = FastAPI(
    title="Energy Prediction API",
    version="1.0.0"
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