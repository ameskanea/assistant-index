from fastapi import FastAPI, Request, status, HTTPException, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import httpx
import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uuid
import tempfile
import openai


from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import pinecone

# Chargement des variables d'environnement
load_dotenv()

# ... (après load_dotenv())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("assistant_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("assistant_api")

# --- Configuration et Initialisation pour l'environnement STABLE ---

# Configuration OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY doit être configurée")

# Initialisation du client OpenAI
openai.api_key = OPENAI_API_KEY

# Configuration Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "assistant-index")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")

# Extraire le nom d'hôte de l'URL Pinecone
import urllib.parse
if PINECONE_HOST:
    parsed_url = urllib.parse.urlparse(PINECONE_HOST)
    PINECONE_HOST = parsed_url.hostname or PINECONE_HOST

# Initialisation de Pinecone avec la nouvelle API V2
try:
    # Configuration initiale de Pinecone avec les clés d'API et l'environnement
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    
    # Vérification de l'existence de l'index ou création si nécessaire
    existing_indexes = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"Création de l'index Pinecone '{PINECONE_INDEX_NAME}'...")
        
        # Création de l'index avec les paramètres spécifiés
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=int(os.getenv("EMBEDDING_MODEL_DIMENSION", 1536)),
            metric="cosine",
            spec=pinecone.ServerlessSpec(
                cloud="aws",
                region=PINECONE_REGION
            )
        )
        logger.info("Index créé avec succès.")
    else:
        logger.info(f"L'index '{PINECONE_INDEX_NAME}' existe déjà.")
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation de Pinecone: {str(e)}")
    logger.warning("Le système continuera à fonctionner sans la fonctionnalité de recherche vectorielle.")
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation de Pinecone: {str(e)}")
    raise

# Initialisation des composants LangChain (version stable)
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except ImportError:
    logger.warning("Using deprecated OpenAIEmbeddings from langchain_community")
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.chat_models import ChatOpenAI

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
chat_model = ChatOpenAI(
    temperature=0.7, 
    model_name=os.getenv("CHAT_MODEL_NAME", "gpt-4"), 
    openai_api_key=OPENAI_API_KEY
)

# === COLLEZ CE NOUVEAU BLOC À LA PLACE DE L'ANCIEN ===
try:
    # 1. Se connecter à l'index Pinecone en utilisant le client Pinecone déjà initialisé (pc)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # 2. Créer le vectorstore en passant l'objet index et l'embedding
    vectorstore = PineconeVectorStore(
        index=pinecone_index,
        embedding=embeddings,
        text_key="text"
    )
    logger.info(f"Connexion à l'index Pinecone '{PINECONE_INDEX_NAME}' réussie.")

except Exception as e:
    logger.error(f"Erreur lors de la connexion à l'index Pinecone: {str(e)}")
    # Initialisation avec une valeur nulle si l'index n'est pas disponible
    vectorstore = None
# ==========================================================

# Création de la chaîne de conversation
qa_chain = None
if vectorstore is not None:
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
    except Exception as e:
        logger.error(f"Erreur lors de la création de la chaîne de conversation: {str(e)}")

# --- Fin de la section d'initialisation ---

# Configuration du rate limiter
limiter = Limiter(key_func=get_remote_address)



# Création de l'application FastAPI
app = FastAPI(
    title="Assistant IA Livres Gourmands",
    description="API FastAPI pour la popup assistant IA du site e-commerce. Utilise OpenAI et Pinecone pour générer des réponses contextuelles.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration du rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),  # Plus sécurisé en production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Constantes
VALID_ROLES = ["admin", "client", "assistant", "guest"]
MAX_MESSAGE_LENGTH = 1000
MAX_CONTEXT_LENGTH = 2000
DEFAULT_TIMEOUT = 5.0
MAX_RETRIES = 3

class AssistantRequest(BaseModel):
    """
    Modèle de requête pour l'assistant IA avec validation avancée.
    """
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    contexte: Optional[str] = Field(default="")
    role: str = Field(default="guest")
    permissions: List[str] = Field(default_factory=list)
    conversation_id: Optional[str] = Field(default=None)
    
    @validator('role')
    def validate_role(cls, v):
        if v not in VALID_ROLES:
            raise ValueError(f"Le rôle doit être parmi : {', '.join(VALID_ROLES)}")
        return v
    
    @validator('permissions')
    def validate_permissions(cls, v):
        allowed_permissions = ["read", "write", "delete", "admin", "analytics"]
        for perm in v:
            if perm not in allowed_permissions:
                raise ValueError(f"Permission invalide : {perm}")
        return v

async def process_file(file: UploadFile) -> List[str]:
    """
    Traite le fichier uploadé et l'ajoute à la base de connaissances
    """
    # Création d'un fichier temporaire
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Chargement du document selon son type
        if suffix.lower() == '.pdf':
            loader = PyPDFLoader(tmp_path)
        elif suffix.lower() == '.txt':
            loader = TextLoader(tmp_path)
        elif suffix.lower() in ['.doc', '.docx']:
            loader = Docx2txtLoader(tmp_path)
        else:
            raise ValueError(f"Type de fichier non supporté: {suffix}")

        documents = loader.load()
        
        # Découpage du texte
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        # Ajout à Pinecone
        vectorstore.add_documents(chunks)
        
        return [doc.page_content[:200] + "..." for doc in chunks]  # Retourne les débuts des chunks

    finally:
        os.unlink(tmp_path)  # Suppression du fichier temporaire

class AssistantResponse(BaseModel):
    """
    Modèle de réponse structurée
    """
    reply: str
    books: List[Dict[str, Any]]
    role: str
    timestamp: datetime
    request_id: Optional[str] = None

class HealthCheck(BaseModel):
    """
    Modèle pour le health check
    """
    status: str
    api_version: str
    services: Dict[str, str]

class ErrorResponse(BaseModel):
    """
    Modèle pour les réponses d'erreur
    """
    error: str
    status_code: int
    timestamp: datetime
    path: Optional[str] = None

async def make_http_request_with_retry(
    url: str, 
    payload: Dict[str, Any], 
    max_retries: int = MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT
) -> Optional[httpx.Response]:
    """
    Effectue une requête HTTP avec retry et backoff exponentiel
    """
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=3.0)) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response
        except httpx.TimeoutException:
            logger.warning(f"Timeout lors de la requête vers {url} (tentative {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Backoff exponentiel
            else:
                logger.error(f"Échec après {max_retries} tentatives pour {url}")
                return None
        except httpx.HTTPStatusError as e:
            logger.error(f"Erreur HTTP {e.response.status_code} lors de la requête vers {url}: {e.response.text}")
            return None
        except Exception as e:
            logger.exception(f"Exception non gérée lors de la requête vers {url}: {str(e)}")
            return None

def build_enriched_prompt(data: AssistantRequest, books: List[Dict[str, Any]]) -> str:
    """
    Construit un prompt enrichi pour Gemini avec le contexte RAG
    """
    role_descriptions = {
        "admin": "Tu peux proposer des actions CRUD et des analyses détaillées.",
        "client": "Tu aides sur le panier, la commande et les recommandations personnalisées.",
        "assistant": "Tu guides et proposes des livres adaptés aux besoins.",
        "guest": "Tu donnes des informations générales et incites à l'inscription."
    }
    
    permissions_text = f"Permissions actives : {', '.join(data.permissions)}" if data.permissions else "Aucune permission spéciale"
    
    prompt = f"""Tu es un assistant IA expert en livres de cuisine pour la boutique "Livres Gourmands".

Rôle de l'utilisateur : {data.role}
{role_descriptions.get(data.role, "")}
{permissions_text}

Contexte de la conversation : {data.contexte}
Question de l'utilisateur : {data.message}

Instructions :
- Réponds de manière claire, précise et adaptée au rôle de l'utilisateur
- Utilise un ton professionnel mais chaleureux
- Si pertinent, propose des actions concrètes adaptées aux permissions
- Limite ta réponse à 3-4 paragraphes maximum
"""

    if books:
        prompt += "\n\nLivres recommandés à intégrer dans ta réponse :\n"
        for book in books:
            titre = book.get('titre', 'Titre inconnu')
            description = book.get('description', 'Pas de description')
            prix = book.get('prix', 'Prix non disponible')
            prompt += f"• {titre} - {description} ({prix}€)\n"
        prompt += "\nIntègre ces suggestions de manière naturelle dans ta réponse."
    
    return prompt

@app.on_event("startup")
async def startup_event():
    """
    Actions à effectuer au démarrage de l'application
    """
    logger.info("Démarrage de l'Assistant IA Livres Gourmands")
    
    # Vérification des variables d'environnement critiques
    if not OPENAI_API_KEY:
        logger.critical("OPENAI_API_KEY non configurée - Le service ne fonctionnera pas")
    
    if not PINECONE_API_KEY:
        logger.critical("PINECONE_API_KEY non configurée - La recherche contextuelle ne fonctionnera pas")
    
    if not os.getenv("LARAVEL_API_URL"):
        logger.warning("LARAVEL_API_URL non configurée - Utilisation de l'URL par défaut")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Actions à effectuer à l'arrêt de l'application
    """
    logger.info("Arrêt de l'Assistant IA Livres Gourmands")

@app.get("/", response_model=HealthCheck)
async def root():
    """
    Endpoint de vérification de l'état de l'API
    """
    # Vérification des services
    services_status = {
        "openai": "ok" if OPENAI_API_KEY else "missing_key",
        "pinecone": "ok" if PINECONE_API_KEY else "missing_key",
        "laravel": "ok",  # Pourrait être amélioré avec un vrai health check
    }
    
    return HealthCheck(
        status="ok",
        api_version="2.0.0",
        services=services_status
    )

@app.get("/health")
async def health_check():
    """
    Endpoint de health check détaillé
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "services": {
                "openai": {
                    "status": "ok" if OPENAI_API_KEY else "missing_key",
                    "model": os.getenv("CHAT_MODEL_NAME", "gpt-4")
                },
                "pinecone": {
                    "status": "ok" if PINECONE_API_KEY and PINECONE_HOST else "configuration_error",
                    "region": PINECONE_REGION,
                    "index": PINECONE_INDEX_NAME
                },
                "laravel": {
                    "status": "ok" if os.getenv("LARAVEL_API_URL") else "missing_url",
                    "url": os.getenv("LARAVEL_API_URL", "default")
                }
            },
            "system": {
                "logs_writable": os.access(".", os.W_OK),
                "memory_available": True,  # Pourrait être amélioré avec psutil
                "disk_space": os.path.exists(".")
            }
        }
    }
    
    # Si un check échoue, le statut global devient "unhealthy"
    if not all(health_status["checks"].values()):
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/assistant/popup")
async def doc_popup():
    """
    Documentation de l'endpoint POST /assistant/popup
    """
    return {
        "info": "Utilisez POST /assistant/popup avec message et contexte.",
        "exemple_requete": {
            "message": "Je cherche un livre de cuisine végétarienne",
            "contexte": "Page d'accueil",
            "role": "client",
            "permissions": ["read"]
        },
        "roles_disponibles": VALID_ROLES,
        "permissions_disponibles": ["read", "write", "delete", "admin", "analytics"],
        "limites": {
            "message_max_length": MAX_MESSAGE_LENGTH,
            "context_max_length": MAX_CONTEXT_LENGTH,
            "rate_limit": "10 requêtes par minute"
        }
    }

@app.post("/assistant/upload")
@limiter.limit("5/minute")
async def upload_file(
    file: UploadFile = File(...),
    request: Request = None
):
    """
    Endpoint pour uploader des fichiers et les ajouter à la base de connaissances
    """
    try:
        chunks = await process_file(file)
        return {
            "message": f"Fichier {file.filename} traité avec succès",
            "chunks_preview": chunks
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors du traitement du fichier: {str(e)}"
        )

@app.post("/assistant/popup", response_model=AssistantResponse)
@limiter.limit("10/minute")
async def handle_popup(data: AssistantRequest, request: Request):
    """
    Endpoint principal pour gérer les requêtes de l'assistant IA.
    
    - Utilise OpenAI pour la génération de réponses
    - Intègre RAG avec Pinecone pour la recherche contextuelle
    - Gère l'historique des conversations
    """
    # Génération d'un ID de requête pour le suivi
    request_id = str(uuid.uuid4())
    logger.info(f"Nouvelle requête {request_id} - Rôle: {data.role}, Message: {data.message[:50]}...")
    
    # 1. Vérification de la clé API OpenAI
    if not OPENAI_API_KEY:
        logger.critical("Clé API OpenAI non configurée")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service d'IA temporairement indisponible"
        )
    
    # 2. Récupération des suggestions de livres depuis Laravel (RAG)
    books = []
    laravel_url = os.getenv("LARAVEL_API_URL", "http://127.0.0.1:8000/api/assistant/books")
    
    try:
        laravel_payload = {
            "role": data.role,
            "permissions": data.permissions,
            "message": data.message,
            "contexte": data.contexte
        }
        
        response = await make_http_request_with_retry(laravel_url, laravel_payload, timeout=3.0)
        
        if response and response.status_code == 200:
            response_data = response.json()
            if "books" in response_data and isinstance(response_data["books"], list):
                books = response_data["books"]
                logger.info(f"Récupéré {len(books)} livres depuis Laravel pour la requête {request_id}")
        else:
            logger.warning(f"Impossible de récupérer les livres depuis Laravel pour la requête {request_id}")
    
    except Exception as e:
        logger.exception(f"Exception lors de l'appel Laravel pour la requête {request_id}: {str(e)}")
        # On continue sans les livres plutôt que d'échouer complètement
    
    # 3. Construction du prompt enrichi
    enriched_prompt = build_enriched_prompt(data, books)
    
    # Initialisation de la conversation ou récupération de l'historique
    conversation_id = data.conversation_id or str(uuid.uuid4())
    chat_history = []  # Idéalement, récupérer depuis une base de données
    reply_text = "Je suis désolé, je n'ai pas pu générer une réponse. Veuillez réessayer."

    try:
        if qa_chain is not None:
            # Utilisation de LangChain pour générer la réponse avec RAG
            result = qa_chain({
                "question": data.message,
                "chat_history": chat_history
            })
            
            reply_text = result["answer"]
            sources = [doc.page_content for doc in result.get("source_documents", [])]
            
            if sources:
                reply_text += "\n\nSources pertinentes:\n" + "\n".join(
                    f"• {source[:200]}..." for source in sources
                )
        else:
            # Si qa_chain n'est pas disponible, utiliser uniquement le modèle de chat
            messages = [
                {"role": "system", "content": enriched_prompt},
                {"role": "user", "content": data.message}
            ]
            
            chat_completion = await openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Utiliser le modèle par défaut
                messages=messages,
                temperature=0.7,
            )
            
            reply_text = chat_completion.choices[0].message.content
        
        logger.info(f"Réponse générée avec succès pour la requête {request_id}")
    except Exception as e:
        logger.exception(f"Exception lors de la génération de la réponse pour la requête {request_id}: {str(e)}")
    
    # 5. Construction et retour de la réponse
    return AssistantResponse(
        reply=reply_text,
        books=books,
        role=data.role,
        timestamp=datetime.now(),
        request_id=request_id
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Gestionnaire personnalisé pour les exceptions HTTP
    """
    logger.error(f"HTTPException: {exc.status_code} - {exc.detail} - Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code,
            timestamp=datetime.now(),
            path=request.url.path
        ).dict()
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Gestionnaire pour toutes les autres exceptions non gérées
    """
    logger.exception(f"Exception non gérée: {str(exc)} - Path: {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Une erreur interne s'est produite",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            timestamp=datetime.now(),
            path=request.url.path
        ).dict()
    )

# Point d'entrée pour uvicorn quand exécuté directement
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("assistant_popup:app", host="0.0.0.0", port=port, reload=True)