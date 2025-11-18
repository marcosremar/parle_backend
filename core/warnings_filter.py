#!/usr/bin/env python3
"""
Filtro de Warnings - Suprime warnings não críticos
Melhora a experiência do usuário removendo warnings desnecessários
"""

import warnings
import logging
import os


def setup_warnings_filter():
    """Configure warnings filter to suppress non-critical warnings"""

    # Suprimir FutureWarnings específicos
    warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*")
    warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated.*")
    warnings.filterwarnings("ignore", message="The argument `trust_remote_code`.*")
    warnings.filterwarnings("ignore", message="TORCH_NCCL_AVOID_RECORD_STREAMS.*")

    # Suprimir warnings de dropout em RNN
    warnings.filterwarnings("ignore", message="dropout option adds dropout.*", category=UserWarning)

    # Suprimir warnings de weight_norm deprecation
    warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

    # Suprimir warnings de vLLM específicos
    warnings.filterwarnings("ignore", message="VLLM_ATTENTION_BACKEND.*")
    warnings.filterwarnings("ignore", message=".*async output processing.*")
    warnings.filterwarnings("ignore", message=".*destroy_process_group.*")

    # Configurar nível de log para reduzir spam
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    # Desabilitar alguns logs verbose se env var estiver setada
    if os.getenv("ULTRAVOX_QUIET", "false").lower() == "true":
        logging.getLogger("vllm").setLevel(logging.ERROR)
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def clean_exit_handlers():
    """Clean up resources on exit to avoid warnings"""

    import atexit
    import torch

    def cleanup_torch():
        try:
            # Clean up CUDA context if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except (RuntimeError, AttributeError):
            pass

    # Register cleanup
    atexit.register(cleanup_torch)


# Auto-setup on import
setup_warnings_filter()
clean_exit_handlers()