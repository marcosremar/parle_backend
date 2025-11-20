"""FastAPI app for User Service - Minimal version for Nomad testing"""
from fastapi import FastAPI
from datetime import datetime

# Create FastAPI app
app = FastAPI(title="User Service", version="1.0.0")

# Health endpoint
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "user",
        "timestamp": datetime.now().isoformat()
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "user-service",
        "version": "1.0.0",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", "8200"))
    uvicorn.run(app, host="0.0.0.0", port=port)

