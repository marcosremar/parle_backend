"""
Scenarios Service Standalone - Consolidated for Nomad deployment
"""
import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, APIRouter
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import sqlite3
import logging
import json
import uuid
from loguru import logger

# Add project root to path for src imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import src modules (fallback to local if not available)
try:
    from .utils.route_helpers import add_standard_endpoints
    from .utils.metrics import increment_metric, set_gauge
except ImportError:
    # Fallback implementations for standalone mode
    def increment_metric(name, value=1, labels=None):
        pass

    def set_gauge(name, value, labels=None):
        pass

    def add_standard_endpoints(router, service_instance=None, service_name=None):
        pass

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "scenarios",
        "port": 8700,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    },
    "database": {
        "path": "/tmp/scenarios.db",
        "enable_vector_search": False  # Disabled for standalone mode
    }
}

def get_config():
    """Get scenarios service configuration"""
    config = DEFAULT_CONFIG.copy()
    return config

# ============================================================================
# Pydantic Models (Standalone)
# ============================================================================

class ScenarioType(str, Enum):
    """Types of conversation scenarios"""
    LANGUAGE_LEARNING = "language_learning"
    CUSTOMER_SERVICE = "customer_service"
    INTERVIEW = "interview"
    CASUAL = "casual"
    CUSTOM = "custom"


class CEFRLevel(str, Enum):
    """Common European Framework of Reference for Languages levels"""
    A1 = "A1"  # Beginner
    A2 = "A2"  # Elementary
    B1 = "B1"  # Intermediate
    B2 = "B2"  # Upper Intermediate
    C1 = "C1"  # Advanced
    C2 = "C2"  # Mastery


class PedagogicalStrategy(str, Enum):
    """Teaching strategies used in language learning"""
    RECAST = "recast"  # Implicit correction through natural speech
    EXPLICIT = "explicit"  # Explicit error correction and explanation
    SCAFFOLDING = "scaffolding"  # Provide hints/support for self-correction
    PROMPTING = "prompting"  # Encourage expansion through questions
    METALINGUISTIC = "metalinguistic"  # Explain language rules directly


class ConversationalStrategy(str, Enum):
    """Conversational strategies for controlling conversation flow and engagement"""
    KEEP_IN_SCENARIO = "keep_in_scenario"  # Keep student focused on scenario context
    REDIRECT_TO_TOPIC = "redirect_to_topic"  # Redirect off-topic responses back to scenario
    MAINTAIN_ENGAGEMENT = "maintain_engagement"  # Encourage continued participation
    ENCOURAGE_EXPANSION = "encourage_expansion"  # Prompt for more detailed responses
    VALIDATE_RESPONSE = "validate_response"  # Acknowledge and validate student response
    BRIDGE_TO_OBJECTIVE = "bridge_to_objective"  # Guide toward learning objective


class StrategiesByLevel(BaseModel):
    """Strategies organized by CEFR level"""
    A1: List[PedagogicalStrategy] = Field(default_factory=list)
    A2: List[PedagogicalStrategy] = Field(default_factory=list)
    B1: List[PedagogicalStrategy] = Field(default_factory=list)
    B2: List[PedagogicalStrategy] = Field(default_factory=list)
    C1: List[PedagogicalStrategy] = Field(default_factory=list)
    C2: List[PedagogicalStrategy] = Field(default_factory=list)


class ScenarioCreate(BaseModel):
    """Request model for creating a scenario"""
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    scenario_type: ScenarioType
    cefr_level: Optional[CEFRLevel] = None
    system_prompt: str = Field(..., min_length=1)
    user_prompt_template: str = Field(..., min_length=1)
    strategies: Optional[List[PedagogicalStrategy]] = Field(default_factory=list)
    conversational_strategies: Optional[List[ConversationalStrategy]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ScenarioUpdate(BaseModel):
    """Request model for updating a scenario"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=1, max_length=1000)
    scenario_type: Optional[ScenarioType] = None
    cefr_level: Optional[CEFRLevel] = None
    system_prompt: Optional[str] = Field(None, min_length=1)
    user_prompt_template: Optional[str] = Field(None, min_length=1)
    strategies: Optional[List[PedagogicalStrategy]] = None
    conversational_strategies: Optional[List[ConversationalStrategy]] = None
    metadata: Optional[Dict[str, Any]] = None


class ScenarioResponse(BaseModel):
    """Response model for a scenario"""
    id: str
    title: str
    description: str
    scenario_type: ScenarioType
    cefr_level: Optional[CEFRLevel]
    system_prompt: str
    user_prompt_template: str
    strategies: List[PedagogicalStrategy]
    conversational_strategies: List[ConversationalStrategy]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ScenarioListResponse(BaseModel):
    """Response model for listing scenarios"""
    scenarios: List[ScenarioResponse]
    total_count: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    database_connected: bool
    total_scenarios: int
    timestamp: datetime


class ValidateTurnRequest(BaseModel):
    """Request model for validating a conversation turn"""
    scenario_id: str
    user_message: str
    ai_response: str
    turn_number: int


class ValidateTurnResponse(BaseModel):
    """Response model for validating a conversation turn"""
    is_valid: bool
    score: float
    feedback: str
    suggestions: List[str]


class InitializeScenarioStateRequest(BaseModel):
    """Request model for initializing scenario state"""
    scenario_id: str
    user_context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ScenarioStateResponse(BaseModel):
    """Response model for scenario state"""
    scenario_id: str
    state: Dict[str, Any]
    initialized_at: datetime

# ============================================================================
# SQLite Database Manager (Standalone)
# ============================================================================

class ScenariosDatabase:
    """
    SQLite database for managing scenarios
    Standalone version without external dependencies
    """

    def __init__(self, db_path: str = "/tmp/scenarios.db"):
        """Initialize scenarios database"""
        # Ensure data directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = str(db_file)
        self.conn: Optional[sqlite3.Connection] = None

        print(f"📁 Scenarios database path: {self.db_path}")
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database tables"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row

            # Create scenarios table
            self.conn.execute('''
            CREATE TABLE IF NOT EXISTS scenarios (
                id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                description TEXT NOT NULL,
                scenario_type TEXT NOT NULL,
                    cefr_level TEXT,
                    system_prompt TEXT NOT NULL,
                    user_prompt_template TEXT NOT NULL,
                    strategies TEXT,  -- JSON array
                    conversational_strategies TEXT,  -- JSON array
                    metadata TEXT,  -- JSON object
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')

            # Create indexes for performance
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_scenario_type ON scenarios(scenario_type)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_cefr_level ON scenarios(cefr_level)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON scenarios(created_at)')

            self.conn.commit()
            print("✅ Scenarios database initialized")

        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            self.conn = None

    def is_connected(self) -> bool:
        """Check database connection"""
        return self.conn is not None

    def get_count(self) -> int:
        """Get total number of scenarios"""
        if not self.conn:
            return 0

        try:
            cursor = self.conn.execute('SELECT COUNT(*) FROM scenarios')
            return cursor.fetchone()[0]
        except Exception:
            return 0

    def create_scenario(self, scenario_data: ScenarioCreate) -> ScenarioResponse:
        """Create a new scenario"""
        if not self.conn:
            raise HTTPException(status_code=503, detail="Database not available")

        scenario_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        try:
            self.conn.execute('''
                INSERT INTO scenarios (
                    id, title, description, scenario_type, cefr_level,
                    system_prompt, user_prompt_template, strategies,
                    conversational_strategies, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                scenario_id,
                scenario_data.title,
                scenario_data.description,
                scenario_data.scenario_type.value,
                scenario_data.cefr_level.value if scenario_data.cefr_level else None,
                scenario_data.system_prompt,
                scenario_data.user_prompt_template,
                json.dumps([s.value for s in scenario_data.strategies or []]),
                json.dumps([s.value for s in scenario_data.conversational_strategies or []]),
                json.dumps(scenario_data.metadata or {}),
                now,
                now
            ))

            self.conn.commit()
            return self.get_scenario(scenario_id)

        except Exception as e:
            self.conn.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to create scenario: {str(e)}")

    def get_scenario(self, scenario_id: str) -> ScenarioResponse:
        """Get a scenario by ID"""
        if not self.conn:
            raise HTTPException(status_code=503, detail="Database not available")

        try:
            cursor = self.conn.execute('SELECT * FROM scenarios WHERE id = ?', (scenario_id,))
            row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Scenario not found")

            return self._row_to_scenario_response(row)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get scenario: {str(e)}")

    def list_scenarios(
        self,
        scenario_type: Optional[ScenarioType] = None,
                      cefr_level: Optional[CEFRLevel] = None,
        page: int = 1,
        page_size: int = 20
    ) -> ScenarioListResponse:
        """List scenarios with optional filtering"""
        if not self.conn:
            raise HTTPException(status_code=503, detail="Database not available")

        try:
            # Build query
            query = 'SELECT * FROM scenarios WHERE 1=1'
            params = []

            if scenario_type:
                query += ' AND scenario_type = ?'
                params.append(scenario_type.value)

            if cefr_level:
                query += ' AND cefr_level = ?'
                params.append(cefr_level.value)

            # Add ordering and pagination
            query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
            params.extend([page_size, (page - 1) * page_size])

            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()

            scenarios = [self._row_to_scenario_response(row) for row in rows]

            # Get total count
            count_query = 'SELECT COUNT(*) FROM scenarios WHERE 1=1'
            count_params = []

            if scenario_type:
                count_query += ' AND scenario_type = ?'
                count_params.append(scenario_type.value)

            if cefr_level:
                count_query += ' AND cefr_level = ?'
                count_params.append(cefr_level.value)

            count_cursor = self.conn.execute(count_query, count_params)
            total_count = count_cursor.fetchone()[0]

            return ScenarioListResponse(
                scenarios=scenarios,
                total_count=total_count,
                page=page,
                page_size=page_size
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list scenarios: {str(e)}")

    def update_scenario(self, scenario_id: str, update_data: ScenarioUpdate) -> ScenarioResponse:
        """Update a scenario"""
        if not self.conn:
            raise HTTPException(status_code=503, detail="Database not available")

        # Get current scenario
        current = self.get_scenario(scenario_id)

        # Build update query
        update_fields = []
        params = []

        if update_data.title is not None:
            update_fields.append('title = ?')
            params.append(update_data.title)

        if update_data.description is not None:
            update_fields.append('description = ?')
            params.append(update_data.description)

        if update_data.scenario_type is not None:
            update_fields.append('scenario_type = ?')
            params.append(update_data.scenario_type.value)

        if update_data.cefr_level is not None:
            update_fields.append('cefr_level = ?')
            params.append(update_data.cefr_level.value if update_data.cefr_level else None)

        if update_data.system_prompt is not None:
            update_fields.append('system_prompt = ?')
            params.append(update_data.system_prompt)

        if update_data.user_prompt_template is not None:
            update_fields.append('user_prompt_template = ?')
            params.append(update_data.user_prompt_template)

        if update_data.strategies is not None:
            update_fields.append('strategies = ?')
            params.append(json.dumps([s.value for s in update_data.strategies]))

        if update_data.conversational_strategies is not None:
            update_fields.append('conversational_strategies = ?')
            params.append(json.dumps([s.value for s in update_data.conversational_strategies]))

        if update_data.metadata is not None:
            update_fields.append('metadata = ?')
            params.append(json.dumps(update_data.metadata))

        if not update_fields:
            return current  # No changes

        # Add updated_at
        update_fields.append('updated_at = ?')
        params.append(datetime.now().isoformat())

        # Add scenario_id
        params.append(scenario_id)

        try:
            query = f'UPDATE scenarios SET {", ".join(update_fields)} WHERE id = ?'
            self.conn.execute(query, params)
            self.conn.commit()

            return self.get_scenario(scenario_id)

        except Exception as e:
            self.conn.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to update scenario: {str(e)}")

    def delete_scenario(self, scenario_id: str) -> bool:
        """Delete a scenario"""
        if not self.conn:
            raise HTTPException(status_code=503, detail="Database not available")

        try:
            cursor = self.conn.execute('DELETE FROM scenarios WHERE id = ?', (scenario_id,))
            deleted = cursor.rowcount > 0
            self.conn.commit()
            return deleted

        except Exception as e:
            self.conn.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to delete scenario: {str(e)}")

    def _row_to_scenario_response(self, row) -> ScenarioResponse:
        """Convert database row to ScenarioResponse"""
        return ScenarioResponse(
            id=row['id'],
            title=row['title'],
            description=row['description'],
            scenario_type=ScenarioType(row['scenario_type']),
                    cefr_level=CEFRLevel(row['cefr_level']) if row['cefr_level'] else None,
            system_prompt=row['system_prompt'],
            user_prompt_template=row['user_prompt_template'],
            strategies=[PedagogicalStrategy(s) for s in json.loads(row['strategies'] or '[]')],
            conversational_strategies=[ConversationalStrategy(s) for s in json.loads(row['conversational_strategies'] or '[]')],
            metadata=json.loads(row['metadata'] or '{}'),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )

    def load_sample_scenarios(self):
        """Load sample scenarios for testing"""
        sample_scenarios = [
            {
                "title": "Ordering Food at a Restaurant",
                "description": "Practice ordering food and interacting with restaurant staff",
                "scenario_type": ScenarioType.LANGUAGE_LEARNING,
                "cefr_level": CEFRLevel.A2,
                "system_prompt": "You are a friendly restaurant server in a casual Italian restaurant. Help the customer practice English while ordering food.",
                "user_prompt_template": "You are at Mario's Italian Restaurant. Practice ordering food in English. Start by greeting the server.",
                "strategies": [PedagogicalStrategy.RECAST, PedagogicalStrategy.SCAFFOLDING],
                "conversational_strategies": [ConversationalStrategy.KEEP_IN_SCENARIO, ConversationalStrategy.ENCOURAGE_EXPANSION],
                "metadata": {"language": "en", "difficulty": "beginner", "duration": 15}
            },
            {
                "title": "Job Interview - Software Developer",
                "description": "Practice common job interview questions for software development positions",
                "scenario_type": ScenarioType.INTERVIEW,
                "cefr_level": CEFRLevel.B2,
                "system_prompt": "You are an experienced interviewer conducting a technical job interview for a software developer position. Ask relevant questions and provide constructive feedback.",
                "user_prompt_template": "You are interviewing for a software developer position at TechCorp. Answer the interviewer's questions professionally and showcase your technical skills.",
                "strategies": [PedagogicalStrategy.EXPLICIT, PedagogicalStrategy.METALINGUISTIC],
                "conversational_strategies": [ConversationalStrategy.VALIDATE_RESPONSE, ConversationalStrategy.BRIDGE_TO_OBJECTIVE],
                "metadata": {"field": "technology", "level": "senior", "duration": 30}
            },
            {
                "title": "Customer Service - Product Return",
                "description": "Handle customer inquiries about returning a defective product",
                "scenario_type": ScenarioType.CUSTOMER_SERVICE,
                "cefr_level": CEFRLevel.B1,
                "system_prompt": "You are a customer service representative for an electronics store. Help customers with returns and refunds professionally.",
                "user_prompt_template": "You bought a laptop from our store but it has issues. Contact customer service to arrange a return and get a refund.",
                "strategies": [PedagogicalStrategy.PROMPTING, PedagogicalStrategy.EXPLICIT],
                "conversational_strategies": [ConversationalStrategy.REDIRECT_TO_TOPIC, ConversationalStrategy.MAINTAIN_ENGAGEMENT],
                "metadata": {"department": "electronics", "issue_type": "returns", "duration": 20}
            }
        ]

        loaded_count = 0
        for scenario_data in sample_scenarios:
            try:
                scenario = ScenarioCreate(**scenario_data)
                self.create_scenario(scenario)
                loaded_count += 1
                print(f"✅ Loaded sample scenario: {scenario.title}")
            except Exception as e:
                print(f"⚠️  Failed to load scenario {scenario_data['title']}: {e}")

        return loaded_count

# ============================================================================
# Global Database Instance
# ============================================================================

try:
    config = get_config()
    scenarios_db = ScenariosDatabase(db_path=config["database"]["path"])
    print("✅ Scenarios Database initialized")
except Exception as e:
    print(f"⚠️  Scenarios Database failed: {e}")
    scenarios_db = None

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Scenarios Service", version="1.0.0")

# ============================================================================
# Routes
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    is_connected = scenarios_db.is_connected() if scenarios_db else False
    total_count = scenarios_db.get_count() if scenarios_db and is_connected else 0

    return HealthResponse(
        status="healthy" if is_connected else "unhealthy",
        database_connected=is_connected,
        total_scenarios=total_count,
        timestamp=datetime.now()
    )

@app.post("/scenarios", response_model=ScenarioResponse)
async def create_scenario(scenario: ScenarioCreate):
    """Create a new scenario"""
    if not scenarios_db:
        raise HTTPException(status_code=503, detail="Database not available")

    return scenarios_db.create_scenario(scenario)

@app.get("/scenarios/{scenario_id}", response_model=ScenarioResponse)
async def get_scenario(scenario_id: str):
    """Get a scenario by ID"""
    if not scenarios_db:
        raise HTTPException(status_code=503, detail="Database not available")

    return scenarios_db.get_scenario(scenario_id)

@app.get("/scenarios", response_model=ScenarioListResponse)
async def list_scenarios(
    scenario_type: Optional[ScenarioType] = None,
    cefr_level: Optional[CEFRLevel] = None,
    page: int = 1,
    page_size: int = 20
):
    """List scenarios with optional filtering"""
    if not scenarios_db:
        raise HTTPException(status_code=503, detail="Database not available")

    if page < 1:
        raise HTTPException(status_code=400, detail="Page must be >= 1")

    if page_size < 1 or page_size > 100:
        raise HTTPException(status_code=400, detail="Page size must be between 1 and 100")

    return scenarios_db.list_scenarios(
        scenario_type=scenario_type,
        cefr_level=cefr_level,
        page=page,
        page_size=page_size
    )

@app.put("/scenarios/{scenario_id}", response_model=ScenarioResponse)
async def update_scenario(scenario_id: str, update_data: ScenarioUpdate):
    """Update a scenario"""
    if not scenarios_db:
        raise HTTPException(status_code=503, detail="Database not available")

    return scenarios_db.update_scenario(scenario_id, update_data)

@app.delete("/scenarios/{scenario_id}")
async def delete_scenario(scenario_id: str):
    """Delete a scenario"""
    if not scenarios_db:
        raise HTTPException(status_code=503, detail="Database not available")

    deleted = scenarios_db.delete_scenario(scenario_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Scenario not found")

    return {"message": "Scenario deleted successfully"}

@app.post("/scenarios/{scenario_id}/validate-turn", response_model=ValidateTurnResponse)
async def validate_turn(scenario_id: str, request: ValidateTurnRequest):
    """Validate a conversation turn (placeholder implementation)"""
    if not scenarios_db:
        raise HTTPException(status_code=503, detail="Database not available")

    # Get the scenario to validate against
    scenario = scenarios_db.get_scenario(scenario_id)

    # Simple validation logic (placeholder)
    is_valid = len(request.user_message.strip()) > 0 and len(request.ai_response.strip()) > 0
    score = 0.8 if is_valid else 0.3

    feedback = "Good conversation flow" if is_valid else "Please provide more detailed responses"
    suggestions = [
        "Try to stay in character",
        "Use more complete sentences",
        "Ask questions to continue the conversation"
    ] if not is_valid else []

    return ValidateTurnResponse(
        is_valid=is_valid,
        score=score,
        feedback=feedback,
        suggestions=suggestions
    )

@app.post("/scenarios/{scenario_id}/initialize-state", response_model=ScenarioStateResponse)
async def initialize_scenario_state(scenario_id: str, request: InitializeScenarioStateRequest):
    """Initialize scenario state (placeholder implementation)"""
    if not scenarios_db:
        raise HTTPException(status_code=503, detail="Database not available")

    # Get the scenario
    scenario = scenarios_db.get_scenario(scenario_id)

    # Initialize basic state
    state = {
        "scenario_id": scenario_id,
        "turn_count": 0,
        "user_context": request.user_context or {},
        "conversation_history": [],
        "objectives_completed": [],
        "current_objective": None
    }

    return ScenarioStateResponse(
        scenario_id=scenario_id,
        state=state,
        initialized_at=datetime.now()
    )

@app.post("/load-sample-scenarios")
async def load_sample_scenarios():
    """Load sample scenarios for testing"""
    if not scenarios_db:
        raise HTTPException(status_code=503, detail="Database not available")

    loaded_count = scenarios_db.load_sample_scenarios()
    return {"message": f"Loaded {loaded_count} sample scenarios"}

# Add standard endpoints
# Create a minimal service instance for health checks
class MinimalService:
    async def health_check(self):
        return {"status": "healthy", "service": "scenarios"}
    def get_service_info(self):
        return {"service": "scenarios", "version": "1.0.0", "status": "running"}

minimal_service = MinimalService()
router = APIRouter()
add_standard_endpoints(router, minimal_service, "scenarios")
app.include_router(router)

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service"""
    print("🚀 Initializing Scenarios Service...")
    print(f"   Database Path: {config['database']['path']}")

    if scenarios_db and scenarios_db.is_connected():
        count = scenarios_db.get_count()
        print(f"   Total Scenarios: {count}")

        # Load sample scenarios if database is empty
        if count == 0:
            print("   Loading sample scenarios...")
            sample_count = scenarios_db.load_sample_scenarios()
            print(f"   ✅ Loaded {sample_count} sample scenarios")
    else:
        print("   ❌ Database connection failed")

    print("✅ Scenarios Service initialized successfully!")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8700"))
    print(f"Starting Scenarios Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)