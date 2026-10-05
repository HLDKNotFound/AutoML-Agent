"""
FastAPI Main Application Entry Point.
Initializes middleware, CORS, routes, and startup checks.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.endpoints import router
from backend.app.core.config import config

app = FastAPI(
    title="ML Automation Agent API",
    description="Automated end-to-end Machine Learning pipeline with LangGraph orchestration, scikit-learn, and Gemini 2.5 Flash.",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "service": "ML Automation Agent",
        "llm_model": config.llm_model,
        "random_state": config.random_state
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=config.api_host, port=config.api_port, reload=True)
