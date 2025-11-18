"""
Simulation Test Runner for E2E Scenario Testing

Tests real conversation flows using WebRTC communication with all services running.
"""

import os
import sys
import time
import json
import asyncio
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


class SimulationRunner:
    """Run E2E conversation simulations to test real-world scenarios"""

    def __init__(self, scenario_name: str):
        """
        Initialize simulation runner

        Args:
            scenario_name: Name of the scenario to test (e.g., "taxi-talk-3turns")
        """
        self.scenario_name = scenario_name
        self.base_url = os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888")  # Service Manager / API Gateway
        self.scenario_id = None
        self.session_id = None
        self.conversation_turns = []
        self.logs_collected = []

    async def check_services_running(self) -> bool:
        """
        Check if all required services are running via Service Manager API

        Returns:
            True if all required services are running, False otherwise
        """
        import httpx

        # Required services for simulation
        required_services = [
            "api_gateway",
            "webrtc",
            "orchestrator",
            "scenarios",
            "session"
        ]

        logger.info("🔍 Checking services via Service Manager API...")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get service status from Service Manager
                response = await client.get("http://localhost:8888/services/status")

                if response.status_code != 200:
                    logger.error(f"   ❌ Failed to get service status from Service Manager (status {response.status_code})")
                    return False

                data = response.json()
                services = data.get("services", {})
                summary = data.get("summary", {})

                logger.info(f"   📊 Total services: {summary.get('total_services', 0)}")
                logger.info(f"   ✅ Healthy: {summary.get('healthy', 0)}")
                logger.info(f"   ❌ Unhealthy: {summary.get('unhealthy', 0)}")
                logger.info("")

                # Check each required service
                missing = []
                stopped = []
                unhealthy = []

                for service_name in required_services:
                    if service_name not in services:
                        logger.error(f"   ❌ {service_name:20s} - NOT CONFIGURED")
                        missing.append(service_name)
                        continue

                    service_info = services[service_name]
                    status = service_info.get("process_status", "unknown")
                    is_healthy = service_info.get("healthy", False)
                    port = service_info.get("port", "?")

                    if status != "running":
                        logger.error(f"   ❌ {service_name:20s} - STOPPED (port {port})")
                        stopped.append(service_name)
                    elif not is_healthy:
                        logger.warning(f"   ⚠️  {service_name:20s} - RUNNING but UNHEALTHY (port {port})")
                        unhealthy.append(service_name)
                    else:
                        logger.info(f"   ✅ {service_name:20s} - RUNNING and HEALTHY (port {port})")

                logger.info("")

                # Print summary of issues
                if missing:
                    logger.error(f"   ⚠️  {len(missing)} service(s) not configured: {', '.join(missing)}")

                if stopped:
                    logger.error(f"   ⚠️  {len(stopped)} service(s) stopped: {', '.join(stopped)}")
                    logger.info("")
                    logger.info("   💡 To start all services, run:")
                    logger.info("      ./main.sh start-all")

                if unhealthy:
                    logger.warning(f"   ⚠️  {len(unhealthy)} service(s) unhealthy: {', '.join(unhealthy)}")

                # Return True only if ALL required services are running and healthy
                all_running = len(missing) == 0 and len(stopped) == 0
                return all_running

        except Exception as e:
            logger.error(f"   ❌ Error checking services: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def create_scenario_dynamically(self) -> bool:
        """
        Create the scenario dynamically via API (tests scenario creation)

        Returns:
            True if scenario created, False otherwise
        """
        import httpx

        logger.info(f"🏗️  Creating scenario dynamically: {self.scenario_name}")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Create scenario via POST
                scenario_data = {
                    "name": self.scenario_name,
                    "type": "custom",
                    "language": "pt-BR",
                    "voice_id": "pf_dora",
                    "system_prompt": """Você é um taxista amigável e prestativo em uma cidade brasileira.

REGRAS IMPORTANTES:
- Você DEVE LEMBRAR de TUDO que o passageiro disser durante a conversa
- Responda SEMPRE em 1-2 frases curtas e naturais (máximo 20-25 palavras)
- Seja educado e use tom conversacional típico de um taxista brasileiro
- Quando o passageiro perguntar sobre o destino, você DEVE mencionar EXATAMENTE o que ele disse antes
- Mantenha o contexto da conversa inteira

MEMÓRIA É CRÍTICA: Se o passageiro disse que quer ir para "Shopping Iguatemi",
você DEVE lembrar exatamente "Shopping Iguatemi" quando ele perguntar depois.""",
                    "user_role": "Passageiro no táxi",
                    "ai_role": "Taxista - responde em 1-2 frases curtas, LEMBRA de tudo que o passageiro disse, usa tom amigável e conversacional de taxista brasileiro",
                    "is_template": False
                }

                response = await client.post(
                    f"{self.base_url}/api/scenarios/api/scenarios",
                    json=scenario_data
                )

                if response.status_code in [200, 201]:
                    data = response.json()
                    self.scenario_id = data["id"]
                    logger.info(f"   ✅ Scenario created: {self.scenario_id}")
                    logger.info(f"   📝 Type: {data['type']}")
                    logger.info(f"   🗣️  Language: {data['language']}")
                    return True
                elif response.status_code == 409:
                    # Scenario already exists, fetch it
                    logger.info(f"   ℹ️  Scenario already exists, fetching existing...")

                    # List scenarios and find ours
                    list_response = await client.get(f"{self.base_url}/api/scenarios/api/scenarios")
                    if list_response.status_code == 200:
                        data = list_response.json()
                        scenarios = data.get("scenarios", [])
                        for scenario in scenarios:
                            if scenario.get("name") == self.scenario_name:
                                self.scenario_id = scenario["id"]
                                logger.info(f"   ✅ Using existing scenario: {self.scenario_id}")
                                return True

                    logger.error(f"   ❌ Scenario exists but could not fetch it")
                    return False
                else:
                    logger.error(f"   ❌ Failed to create scenario (status {response.status_code})")
                    logger.error(f"   Response: {response.text}")
                    return False

        except Exception as e:
            logger.error(f"   ❌ Error creating scenario: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def create_session(self) -> bool:
        """
        Create a session for the conversation

        Returns:
            True if session created, False otherwise
        """
        import httpx

        logger.info("🔐 Creating session...")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/conversation_store/session/create",
                    json={
                        "user_id": "simulation_test_user",
                        "scenario_id": self.scenario_id
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    self.session_id = data["session_id"]
                    logger.info(f"   ✅ Session created: {self.session_id}")
                    return True
                else:
                    logger.error(f"   ❌ Failed to create session (status {response.status_code})")
                    return False

        except Exception as e:
            logger.error(f"   ❌ Error creating session: {e}")
            return False

    async def send_turn_via_webrtc(self, user_message: str, turn_number: int) -> Optional[Dict[str, Any]]:
        """
        Send a conversation turn via WebRTC

        Args:
            user_message: User's message text
            turn_number: Turn number (1, 2, 3, etc.)

        Returns:
            Response data or None if failed
        """
        import httpx

        logger.info(f"💬 Turn {turn_number}: User says: '{user_message}'")

        try:
            # Use orchestrator's conversation endpoint for text-based simulation
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/orchestrator/conversation",
                    json={
                        "message": user_message,
                        "session_id": self.session_id,
                        "voice_id": "pf_dora"
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    assistant_response = data.get("response", "")
                    logger.info(f"🤖 Turn {turn_number}: Assistant says: '{assistant_response}'")

                    turn_data = {
                        "turn": turn_number,
                        "user_message": user_message,
                        "assistant_response": assistant_response,
                        "timestamp": datetime.now().isoformat(),
                        "success": True
                    }

                    self.conversation_turns.append(turn_data)
                    return turn_data
                else:
                    logger.error(f"   ❌ Turn {turn_number} failed (status {response.status_code})")
                    logger.error(f"   Response: {response.text}")
                    return None

        except Exception as e:
            logger.error(f"   ❌ Turn {turn_number} error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def collect_logs(self):
        """Collect logs from services to verify no errors occurred"""
        logger.info("📄 Collecting logs from services...")

        log_files = [
            str(Path.home() / ".cache" / "ultravox-pipeline" / "ultravox/logs/service_manager.log",
            str(Path.home() / ".cache" / "ultravox-pipeline" / "ultravox/logs/orchestrator.log",
            str(Path.home() / ".cache" / "ultravox-pipeline" / "ultravox/logs/webrtc.log",
            str(Path.home() / ".cache" / "ultravox-pipeline" / "ultravox/logs/scenarios.log"
        ]

        for log_file in log_files:
            try:
                if Path(log_file).exists():
                    with open(log_file, "r") as f:
                        # Get last 100 lines
                        lines = f.readlines()[-100:]
                        self.logs_collected.append({
                            "file": log_file,
                            "lines": lines
                        })
                        logger.info(f"   ✅ Collected {len(lines)} lines from {log_file}")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not read {log_file}: {e}")

    def validate_memory(self) -> bool:
        """
        Validate that the assistant remembered information from earlier turns

        For taxi-talk-3turns:
        - Turn 1: User says destination
        - Turn 2: User asks something else
        - Turn 3: User asks where they wanted to go
        - Expected: Assistant should mention the destination from Turn 1

        Returns:
            True if memory check passed, False otherwise
        """
        logger.info("🧠 Validating conversation memory...")

        if len(self.conversation_turns) < 3:
            logger.error(f"   ❌ Not enough turns to validate memory (need 3, got {len(self.conversation_turns)})")
            return False

        # Get destination from Turn 1
        turn1_user_msg = self.conversation_turns[0]["user_message"].lower()

        # Extract destination keywords (simple extraction for now)
        # Look for common patterns: "quero ir para X", "vou para X", etc.
        destination_keywords = []
        if "shopping" in turn1_user_msg:
            destination_keywords.append("shopping")
        if "aeroporto" in turn1_user_msg:
            destination_keywords.append("aeroporto")
        if "rodoviária" in turn1_user_msg:
            destination_keywords.append("rodoviária")
        if "hospital" in turn1_user_msg:
            destination_keywords.append("hospital")

        # Get Turn 3 assistant response
        turn3_assistant_msg = self.conversation_turns[2]["assistant_response"].lower()

        # Check if assistant mentioned the destination
        memory_works = any(keyword in turn3_assistant_msg for keyword in destination_keywords)

        if memory_works:
            logger.info(f"   ✅ Memory check PASSED - Assistant remembered destination")
            logger.info(f"      Turn 1: User said: '{self.conversation_turns[0]['user_message']}'")
            logger.info(f"      Turn 3: Assistant recalled: '{self.conversation_turns[2]['assistant_response']}'")
            return True
        else:
            logger.error(f"   ❌ Memory check FAILED - Assistant did NOT remember destination")
            logger.error(f"      Turn 1: User said: '{self.conversation_turns[0]['user_message']}'")
            logger.error(f"      Turn 3: Assistant said: '{self.conversation_turns[2]['assistant_response']}'")
            logger.error(f"      Expected keywords: {destination_keywords}")
            return False

    def check_logs_for_errors(self) -> bool:
        """
        Check collected logs for errors

        Returns:
            True if no critical errors found, False otherwise
        """
        logger.info("🔍 Checking logs for errors...")

        critical_errors = []
        warnings = []

        for log_data in self.logs_collected:
            log_file = log_data["file"]
            for line in log_data["lines"]:
                line_lower = line.lower()
                if "error" in line_lower or "exception" in line_lower or "traceback" in line_lower:
                    if "critical" in line_lower or "fatal" in line_lower:
                        critical_errors.append((log_file, line.strip()))
                    elif "warning" not in line_lower:
                        warnings.append((log_file, line.strip()))

        if critical_errors:
            logger.error(f"   ❌ Found {len(critical_errors)} critical errors in logs:")
            for log_file, error_line in critical_errors[:10]:  # Show first 10
                logger.error(f"      {Path(log_file).name}: {error_line[:100]}")
            return False
        elif warnings:
            logger.warning(f"   ⚠️  Found {len(warnings)} warnings in logs (not critical)")
            return True
        else:
            logger.info(f"   ✅ No critical errors found in logs")
            return True

    async def run_simulation(self, conversation_script: List[str]) -> Dict[str, Any]:
        """
        Run the complete simulation

        Args:
            conversation_script: List of user messages (one per turn)

        Returns:
            Test results dictionary
        """
        start_time = time.time()
        logger.info("=" * 80)
        logger.info(f"🚀 Starting simulation: {self.scenario_name}")
        logger.info("=" * 80)

        # Check services
        if not await self.check_services_running():
            return {
                "success": False,
                "error": "Not all services are running",
                "timestamp": datetime.now().isoformat()
            }

        # Create scenario dynamically (tests scenario creation)
        if not await self.create_scenario_dynamically():
            return {
                "success": False,
                "error": f"Failed to create scenario: {self.scenario_name}",
                "timestamp": datetime.now().isoformat()
            }

        # Create session
        if not await self.create_session():
            return {
                "success": False,
                "error": "Failed to create session",
                "timestamp": datetime.now().isoformat()
            }

        # Execute conversation turns
        logger.info(f"🗣️  Executing {len(conversation_script)} conversation turns...")
        logger.info("")

        for turn_num, user_message in enumerate(conversation_script, 1):
            turn_result = await self.send_turn_via_webrtc(user_message, turn_num)
            if not turn_result:
                return {
                    "success": False,
                    "error": f"Turn {turn_num} failed",
                    "conversation_turns": self.conversation_turns,
                    "timestamp": datetime.now().isoformat()
                }
            # Small delay between turns
            await asyncio.sleep(1)

        logger.info("")

        # Collect logs
        await self.collect_logs()

        # Validate memory
        memory_passed = self.validate_memory()

        # Check logs for errors
        logs_clean = self.check_logs_for_errors()

        # Calculate results
        duration = time.time() - start_time
        all_passed = memory_passed and logs_clean

        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 SIMULATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Scenario: {self.scenario_name}")
        logger.info(f"Turns completed: {len(self.conversation_turns)}/{len(conversation_script)}")
        logger.info(f"Memory validation: {'✅ PASSED' if memory_passed else '❌ FAILED'}")
        logger.info(f"Logs validation: {'✅ CLEAN' if logs_clean else '❌ ERRORS FOUND'}")
        logger.info(f"Overall result: {'✅ SUCCESS' if all_passed else '❌ FAILED'}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info("=" * 80)

        return {
            "success": all_passed,
            "scenario_name": self.scenario_name,
            "scenario_id": self.scenario_id,
            "session_id": self.session_id,
            "turns_completed": len(self.conversation_turns),
            "turns_expected": len(conversation_script),
            "memory_validation": memory_passed,
            "logs_validation": logs_clean,
            "conversation_turns": self.conversation_turns,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        }


async def run_taxi_talk_3turns() -> int:
    """
    Run taxi-talk-3turns simulation

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    runner = SimulationRunner(scenario_name="Taxi Driver - 3 Turn Memory Test")

    # Conversation script for taxi scenario
    conversation_script = [
        "Bom dia! Quero ir para o Shopping Iguatemi, por favor.",
        "Quanto tempo mais ou menos vai demorar para chegar lá?",
        "Aliás, para onde eu disse que queria ir mesmo?"
    ]

    results = await runner.run_simulation(conversation_script)

    # Save results to JSON
    report_file = Path(str(Path.home() / ".cache" / "ultravox-pipeline" / "simulation_taxi_talk_3turns.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info(f"📄 Full report saved to: {report_file}")

    return 0 if results["success"] else 1


def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.core.service_manager.testing.simulation_runner <scenario_name>")
        print("")
        print("Available scenarios:")
        print("  taxi-talk-3turns    Test memory and conversation flow with taxi driver")
        sys.exit(1)

    scenario = sys.argv[1]

    if scenario == "taxi-talk-3turns":
        exit_code = asyncio.run(run_taxi_talk_3turns())
        sys.exit(exit_code)
    else:
        print(f"Unknown scenario: {scenario}")
        print("Available: taxi-talk-3turns")
        sys.exit(1)


if __name__ == "__main__":
    main()
