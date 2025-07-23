from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Assistant IA Livres Gourmands",
    description="API FastAPI pour la popup assistant IA du site e-commerce. Utilise Gemini pour générer les réponses.",
    version="1.0.0"
)

# CORS pour le frontend Laravel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AssistantRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Message utilisateur")
    contexte: str = Field(..., min_length=1, description="Contexte de la page ou de l'utilisateur")

@app.get("/")
def root():
    return {"status": "ok", "message": "Assistant IA Livres Gourmands - API FastAPI"}

@app.post("/assistant/popup", response_model=dict, tags=["Assistant"])
async def handle_popup(data: AssistantRequest):
    """
    Reçoit un message utilisateur et un contexte, retourne la réponse générée par Gemini.
    """
    # ... (logique Gemini ici, voir assistant_popup.py)
    return {"reply": "Réponse simulée (Gemini)"}

@app.get("/assistant/popup", tags=["Assistant"])
def doc_popup():
    """
    Endpoint de test GET pour vérifier la route.
    """
    return {"info": "Utilisez POST /assistant/popup avec message et contexte."}

# Pour la documentation interactive :
# Accès à /docs (Swagger UI) et /redoc
