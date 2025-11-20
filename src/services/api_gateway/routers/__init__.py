"""
FastAPI routers for different endpoints
Communication Manager initialization for all routers
"""

from typing import Optional

# Global Communication Manager instance shared by all routers
_comm_manager: Optional['ServiceCommunicationManager'] = None


def initialize_comm_manager_for_routers(comm_manager):
    """
    Initialize Communication Manager for all routers

    This should be called from APIGatewayService.initialize()
    to pass the service's Communication Manager to all routers.
    """
    global _comm_manager
    _comm_manager = comm_manager

    # Set comm_manager in each router module that needs it
    try:
        from . import llm
        if hasattr(llm, 'set_comm_manager'):
            llm.set_comm_manager(comm_manager)
    except ImportError:
        pass

    try:
        from . import tts
        if hasattr(tts, 'set_comm_manager'):
            tts.set_comm_manager(comm_manager)
    except ImportError:
        pass

    try:
        from . import conversation
        if hasattr(conversation, 'set_comm_manager'):
            conversation.set_comm_manager(comm_manager)
    except ImportError:
        pass

    try:
        from . import validate
        if hasattr(validate, 'set_comm_manager'):
            validate.set_comm_manager(comm_manager)
    except ImportError:
        pass

    try:
        from . import models
        if hasattr(models, 'set_comm_manager'):
            models.set_comm_manager(comm_manager)
    except ImportError:
        pass

    try:
        from . import binary
        if hasattr(binary, 'set_comm_manager'):
            binary.set_comm_manager(comm_manager)
    except ImportError:
        pass

    try:
        from . import scenarios
        if hasattr(scenarios, 'set_comm_manager'):
            scenarios.set_comm_manager(comm_manager)
    except ImportError:
        pass

    try:
        from . import session
        if hasattr(session, 'set_comm_manager'):
            session.set_comm_manager(comm_manager)
    except ImportError:
        pass