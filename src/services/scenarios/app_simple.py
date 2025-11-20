"""
Simple Scenarios Service for testing
"""
import uvicorn
import os
import sqlite3
from fastapi import FastAPI, HTTPException
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel
from enum import Enum

# Basic models
class ScenarioType(str, Enum):
    LANGUAGE_LEARNING = "language_learning"
    CUSTOMER_SERVICE = "customer_service"
    INTERVIEW = "interview"
    CASUAL = "casual"
    CUSTOM = "custom"

class ScenarioResponse(BaseModel):
    id: str
    title: str
    description: str
    scenario_type: ScenarioType
    created_at: datetime

# Simple database
class SimpleScenariosDB:
    def __init__(self, db_path="/tmp/scenarios.db"):
        self.db_path = db_path
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS scenarios (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    scenario_type TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            self.conn.commit()
            print("✅ Database initialized")
        except Exception as e:
            print(f"❌ DB init failed: {e}")
            self.conn = None
    
    def is_connected(self):
        return self.conn is not None
    
    def get_count(self):
        if not self.conn:
            return 0
        try:
            cursor = self.conn.execute('SELECT COUNT(*) FROM scenarios')
            return cursor.fetchone()[0]
        except:
            return 0
    
    def create_scenario(self, title, description, scenario_type):
        if not self.conn:
            raise HTTPException(status_code=503, detail="DB not available")
        
        scenario_id = str(__import__('uuid').uuid4())
        now = datetime.now().isoformat()
        
        try:
            self.conn.execute(
                'INSERT INTO scenarios VALUES (?, ?, ?, ?, ?)',
                (scenario_id, title, description, scenario_type, now)
            )
            self.conn.commit()
            return ScenarioResponse(
                id=scenario_id,
                title=title,
                description=description,
                scenario_type=ScenarioType(scenario_type),
                created_at=datetime.fromisoformat(now)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Create failed: {str(e)}")
    
    def list_scenarios(self):
        if not self.conn:
            raise HTTPException(status_code=503, detail="DB not available")
        
        try:
            cursor = self.conn.execute('SELECT * FROM scenarios ORDER BY created_at DESC')
            rows = cursor.fetchall()
            return [
                ScenarioResponse(
                    id=row[0],
                    title=row[1],
                    description=row[2],
                    scenario_type=ScenarioType(row[3]),
                    created_at=datetime.fromisoformat(row[4])
                ) for row in rows
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")

# Global DB
db = SimpleScenariosDB()

# FastAPI app
app = FastAPI(title="Scenarios Service", version="1.0.0")

@app.get("/health")
async def health():
    return {
        "status": "healthy" if db.is_connected() else "unhealthy",
        "database_connected": db.is_connected(),
        "total_scenarios": db.get_count(),
        "timestamp": datetime.now()
    }

@app.post("/scenarios")
async def create_scenario(title: str, description: str, scenario_type: ScenarioType):
    return db.create_scenario(title, description, scenario_type.value)

@app.get("/scenarios")
async def list_scenarios():
    return {"scenarios": db.list_scenarios()}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8700"))
    print(f"Starting Scenarios Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
