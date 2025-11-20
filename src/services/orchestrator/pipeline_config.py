"""
Pipeline Configuration and Management
Defines the two main pipelines and their specific configurations
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path to import ServiceRegistry
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.config.service_config import ServiceRegistry, ServiceType, get_service_port

# Global Communication Manager (can be set by orchestrator service)
comm_manager: Optional['ServiceCommunicationManager'] = None


def set_comm_manager(cm):
    """Set Communication Manager instance from orchestrator service"""
    global comm_manager
    comm_manager = cm


class PipelineType(Enum):
    """Types of pipelines available"""
    WEBRTC_INTERNAL = "webrtc_internal"
    API_EXTERNAL = "api_external"


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    host: str
    port: int
    path: str = "/health"
    required: bool = True
    timeout: float = 5.0

    @property
    def url(self) -> str:
        """Get full URL for the endpoint"""
        return f"http://{self.host}:{self.port}{self.path}"


@dataclass
class PipelineStage:
    """Pipeline stage configuration"""
    name: str
    service: ServiceEndpoint
    next_stages: List[str] = field(default_factory=list)
    processing_type: str = "sequential"  # sequential, parallel, conditional
    fallback_service: Optional[ServiceEndpoint] = None


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    name: str
    type: PipelineType
    description: str
    stages: Dict[str, PipelineStage]
    entry_point: str
    output_stage: str
    enabled: bool = True
    max_latency_ms: int = 1000
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "backoff_ms": 100,
        "max_backoff_ms": 1000
    })
    monitoring: Dict[str, Any] = field(default_factory=lambda: {
        "track_latency": True,
        "track_errors": True,
        "alert_on_failure": True
    })


class PipelineManager:
    """Manages pipeline configurations and operations"""

    def __init__(self):
        self.pipelines = self._initialize_pipelines()
        self.metrics = {
            PipelineType.WEBRTC_INTERNAL: {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_latency_ms": 0,
                "last_execution": None
            },
            PipelineType.API_EXTERNAL: {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_latency_ms": 0,
                "last_execution": None
            }
        }

    def _initialize_pipelines(self) -> Dict[PipelineType, PipelineConfig]:
        """Initialize pipeline configurations"""

        # WebRTC Internal Pipeline Configuration
        webrtc_pipeline = PipelineConfig(
            name="WebRTC Internal Pipeline",
            type=PipelineType.WEBRTC_INTERNAL,
            description="Real-time audio/video communication pipeline via WebRTC",
            entry_point="webrtc_gateway",
            output_stage="tts_service",
            stages={
                "webrtc_gateway": PipelineStage(
                    name="WebRTC Gateway",
                    service=ServiceEndpoint(
                        name="webrtc-gateway",
                        host="localhost",
                        port=get_service_port(ServiceType.WEBRTC_GATEWAY),  # 8500
                        path="/health"
                    ),
                    next_stages=["websocket_relay"],
                    processing_type="sequential"
                ),
                "websocket_relay": PipelineStage(
                    name="WebSocket Relay",
                    service=ServiceEndpoint(
                        name="websocket-gateway",
                        host="localhost",
                        port=get_service_port(ServiceType.WEBSOCKET_GATEWAY),  # 8302
                        path="/health"
                    ),
                    next_stages=["llm_ultravox"],
                    processing_type="sequential"
                ),
                "llm_ultravox": PipelineStage(
                    name="Ultravox LLM",
                    service=ServiceEndpoint(
                        name="llm-service",
                        host="localhost",
                        port=get_service_port(ServiceType.LLM_SERVICE),  # 8100
                        path="/health"
                    ),
                    next_stages=["tts_service"],
                    processing_type="sequential"
                ),
                "tts_service": PipelineStage(
                    name="TTS Service",
                    service=ServiceEndpoint(
                        name="tts-service",
                        host="localhost",
                        port=get_service_port(ServiceType.TTS_SERVICE),  # 8101
                        path="/health"
                    ),
                    next_stages=[],
                    processing_type="sequential"
                )
            },
            max_latency_ms=500,  # Real-time requirement
            monitoring={
                "track_latency": True,
                "track_errors": True,
                "alert_on_failure": True,
                "track_audio_quality": True,
                "track_video_quality": True
            }
        )

        # API External Pipeline Configuration
        api_pipeline = PipelineConfig(
            name="API External Pipeline",
            type=PipelineType.API_EXTERNAL,
            description="REST API based pipeline for external integrations",
            entry_point="api_gateway",
            output_stage="tts_service",
            stages={
                "api_gateway": PipelineStage(
                    name="API Gateway",
                    service=ServiceEndpoint(
                        name="api-gateway",
                        host="localhost",
                        port=get_service_port(ServiceType.API_GATEWAY),  # 8020
                        path="/health"
                    ),
                    next_stages=["llm_processing"],
                    processing_type="sequential"
                ),
                "llm_processing": PipelineStage(
                    name="LLM Processing",
                    service=ServiceEndpoint(
                        name="llm-service",
                        host="localhost",
                        port=get_service_port(ServiceType.LLM_SERVICE),  # 8100
                        path="/health"
                    ),
                    next_stages=["tts_service"],
                    processing_type="parallel",  # Can process with STT in parallel
                    fallback_service=ServiceEndpoint(
                        name="stt",
                        host="localhost",
                        port=get_service_port(ServiceType.STT_SERVICE),  # 8099
                        path="/health",
                        required=False
                    )
                ),
                "stt_service": PipelineStage(
                    name="STT Service",
                    service=ServiceEndpoint(
                        name="stt",
                        host="localhost",
                        port=get_service_port(ServiceType.STT_SERVICE),  # 8099
                        path="/health",
                        required=False  # Optional, as Ultravox handles audio
                    ),
                    next_stages=["llm_processing"],
                    processing_type="sequential"
                ),
                "tts_service": PipelineStage(
                    name="TTS Service",
                    service=ServiceEndpoint(
                        name="tts-service",
                        host="localhost",
                        port=get_service_port(ServiceType.TTS_SERVICE),  # 8101
                        path="/health"
                    ),
                    next_stages=[],
                    processing_type="sequential"
                )
            },
            max_latency_ms=2000,  # More lenient for API calls
            retry_policy={
                "max_retries": 5,
                "backoff_ms": 200,
                "max_backoff_ms": 2000
            }
        )

        return {
            PipelineType.WEBRTC_INTERNAL: webrtc_pipeline,
            PipelineType.API_EXTERNAL: api_pipeline
        }

    def get_pipeline(self, pipeline_type: PipelineType) -> Optional[PipelineConfig]:
        """Get pipeline configuration by type"""
        return self.pipelines.get(pipeline_type)

    def get_all_pipelines(self) -> Dict[PipelineType, PipelineConfig]:
        """Get all pipeline configurations"""
        return self.pipelines

    def get_pipeline_flow(self, pipeline_type: PipelineType) -> str:
        """Get visual representation of pipeline flow"""
        pipeline = self.get_pipeline(pipeline_type)
        if not pipeline:
            return "Pipeline not found"

        flow_parts = []
        current_stage = pipeline.entry_point
        visited = set()

        while current_stage and current_stage not in visited:
            visited.add(current_stage)
            stage = pipeline.stages.get(current_stage)
            if stage:
                flow_parts.append(f"{stage.service.name}({stage.service.port})")
                if stage.next_stages:
                    current_stage = stage.next_stages[0]
                else:
                    break

        return " → ".join(flow_parts)

    def get_pipeline_services(self, pipeline_type: PipelineType) -> List[ServiceEndpoint]:
        """Get all services required for a pipeline"""
        pipeline = self.get_pipeline(pipeline_type)
        if not pipeline:
            return []

        services = []
        for stage in pipeline.stages.values():
            services.append(stage.service)
            if stage.fallback_service:
                services.append(stage.fallback_service)

        # Remove duplicates while preserving order
        seen = set()
        unique_services = []
        for service in services:
            key = (service.host, service.port)
            if key not in seen:
                seen.add(key)
                unique_services.append(service)

        return unique_services

    async def execute_pipeline(self, pipeline_type: PipelineType, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a pipeline with the given data"""
        import httpx
        from datetime import datetime

        pipeline = self.get_pipeline(pipeline_type)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_type} not found")

        if not pipeline.enabled:
            raise ValueError(f"Pipeline {pipeline_type} is disabled")

        start_time = datetime.now()
        result = {"stages_executed": [], "data": data}

        try:
            # Start from entry point
            current_stage = pipeline.entry_point
            stage_data = data

            async with httpx.AsyncClient(timeout=30.0) as client:
                while current_stage:
                    stage = pipeline.stages.get(current_stage)
                    if not stage:
                        break

                    # Execute stage (simplified - in real implementation would call actual service)
                    result["stages_executed"].append({
                        "stage": current_stage,
                        "service": stage.service.name,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Move to next stage
                    if stage.next_stages:
                        current_stage = stage.next_stages[0]
                    else:
                        break

            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics[pipeline_type]["total_executions"] += 1
            self.metrics[pipeline_type]["successful_executions"] += 1
            self.metrics[pipeline_type]["last_execution"] = datetime.now().isoformat()

            # Update average latency
            total = self.metrics[pipeline_type]["total_executions"]
            avg = self.metrics[pipeline_type]["average_latency_ms"]
            self.metrics[pipeline_type]["average_latency_ms"] = ((avg * (total - 1)) + execution_time) / total

            result["execution_time_ms"] = execution_time
            return result

        except Exception as e:
            self.metrics[pipeline_type]["total_executions"] += 1
            self.metrics[pipeline_type]["failed_executions"] += 1
            self.metrics[pipeline_type]["last_execution"] = datetime.now().isoformat()
            raise e

    def get_metrics(self, pipeline_type: PipelineType) -> Dict[str, Any]:
        """Get metrics for a specific pipeline"""
        return self.metrics.get(pipeline_type, {})

    def update_metrics(self, pipeline_type: PipelineType, metrics: Dict[str, Any]):
        """Update pipeline metrics"""
        if pipeline_type not in self.metrics:
            self.metrics[pipeline_type] = {
                "requests": 0,
                "errors": 0,
                "total_latency_ms": 0,
                "last_updated": None
            }

        pipeline_metrics = self.metrics[pipeline_type]
        pipeline_metrics["requests"] += metrics.get("requests", 0)
        pipeline_metrics["errors"] += metrics.get("errors", 0)
        pipeline_metrics["total_latency_ms"] += metrics.get("latency_ms", 0)
        pipeline_metrics["last_updated"] = datetime.now().isoformat()

        # Calculate average latency
        if pipeline_metrics["requests"] > 0:
            pipeline_metrics["avg_latency_ms"] = (
                pipeline_metrics["total_latency_ms"] / pipeline_metrics["requests"]
            )

    def get_metrics(self, pipeline_type: Optional[PipelineType] = None) -> Dict:
        """Get pipeline metrics"""
        if pipeline_type:
            return self.metrics.get(pipeline_type, {})
        return self.metrics

    def validate_pipeline(self, pipeline_type: PipelineType) -> Dict[str, Any]:
        """Validate that all required services for a pipeline are available"""
        pipeline = self.get_pipeline(pipeline_type)
        if not pipeline:
            return {
                "valid": False,
                "error": f"Pipeline {pipeline_type.value} not found"
            }

        validation_result = {
            "valid": True,
            "pipeline": pipeline.name,
            "type": pipeline_type.value,
            "stages": {},
            "warnings": [],
            "errors": []
        }

        for stage_name, stage in pipeline.stages.items():
            stage_status = {
                "name": stage.name,
                "service": stage.service.name,
                "port": stage.service.port,
                "required": stage.service.required,
                "available": False
            }

            # Check if the service is actually available by making a health check
            # Use requests for synchronous health check (GET request)
            import requests
            try:
                health_url = f"http://{stage.service.host}:{stage.service.port}/health"
                response = requests.get(health_url, timeout=2)
                if response.status_code < 500:
                    stage_status["available"] = True
            except (requests.RequestException, ConnectionError, TimeoutError) as e:
                stage_status["available"] = False
                # Log the error for debugging purposes
                from loguru import logger
                logger.debug(f"Health check failed for {stage.service.name}: {e}")

            # Note: Communication Manager is available for actual service calls (POST)
            # Health checks remain as direct HTTP GET requests

            validation_result["stages"][stage_name] = stage_status

            if stage.service.required and not stage_status["available"]:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Required service {stage.service.name} on port {stage.service.port} is not available"
                )
            elif not stage.service.required and not stage_status["available"]:
                validation_result["warnings"].append(
                    f"Optional service {stage.service.name} on port {stage.service.port} is not available"
                )

        return validation_result


# Singleton instance
pipeline_manager = PipelineManager()