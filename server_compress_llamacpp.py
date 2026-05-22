#!/usr/bin/env python3
"""LiteLLM proxy for the llama.cpp GLM backend.

Architecture:
    VS Code / GitHub Copilot plugin -> LiteLLM (12111) -> llama.cpp (12112)
"""

import logging
import os
from pathlib import Path

import litellm

import glm_compress       # noqa: F401 — strips thinking tokens from history
import glm_token_tracker  # noqa: F401 — records per-request token usage

glm_compress.register()
glm_token_tracker.register()

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

litellm_logger = logging.getLogger("litellm.image_request")
litellm_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_DIR / "litellm_llamacpp_image_requests.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
litellm_logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter("%(asctime)s %(name)-25s %(levelname)-8s %(message)s"))
litellm_logger.addHandler(ch)

litellm_logger.info("=" * 60)
litellm_logger.info("LiteLLM llama.cpp proxy started")

os.environ["LITELLM_LOG"] = "DEBUG"
os.environ["LITELLM_REQUEST_LOGGING"] = "true"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-25s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "litellm_llamacpp_detailed.log"),
        logging.StreamHandler(),
    ],
)

logging.getLogger("litellm").setLevel(logging.DEBUG)
logger = logging.getLogger("server_compress_llamacpp")

LITELLM_PORT = os.environ.get("LITE_LLM_PROXY_PORT", "12111")
LITELLM_HOST = os.environ.get("LITE_LLM_PROXY_HOST", "0.0.0.0")
CONFIG_PATH = Path(__file__).parent / "lite_llm_config_llamacpp.yaml"

logger.info("Starting LiteLLM llama.cpp proxy on %s:%s", LITELLM_HOST, LITELLM_PORT)
logger.info("Config: %s", CONFIG_PATH)
logger.info("Assistant history sanitization enabled")

os.environ.pop("LITELLM_MASTER_KEY", None)
os.environ.pop("LITELLM_SALT_KEY", None)
os.environ["CONFIG_FILE_PATH"] = str(CONFIG_PATH)

logger.info("=" * 60)
logger.info("LiteLLM proxy starting in-process with GLM history sanitizer")
logger.info("=" * 60)

from litellm.integrations.custom_logger import CustomLogger

registered_callbacks = [cb for cb in litellm.callbacks if isinstance(cb, CustomLogger)]
logger.info("Registered custom callbacks: %d", len(registered_callbacks))
for cb in registered_callbacks:
    logger.info("  - %s", type(cb).__name__)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "litellm.proxy.proxy_server:app",
        host=LITELLM_HOST,
        port=int(LITELLM_PORT),
        reload=False,
        log_level="debug",
    )
