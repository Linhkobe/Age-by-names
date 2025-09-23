from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.service import service

app = FastAPI(title="Age by First Name")

# Static & templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

class PredictRequest(BaseModel):
    name: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Health endpoints (nice to have) ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    return {"ready": True}

# --- API: both GET (query) and POST (JSON) ---

# GET /predict?name=alice
@app.get("/predict")
def predict_get(name: str):
    """Convenience for browser form submits."""
    return JSONResponse(service.predict(name))

# POST /predict  with {"name":"alice"}
@app.post("/predict")
def predict_post(req: PredictRequest):
    return JSONResponse(service.predict(req.name))
