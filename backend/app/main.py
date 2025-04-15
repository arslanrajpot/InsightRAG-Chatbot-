from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers.api import router as api_router
from .config import Config

app = FastAPI()

# Validate configuration on startup
Config.validate()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")