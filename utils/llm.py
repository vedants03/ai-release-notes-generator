from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging
import openai
import os

load_dotenv()

logger = logging.getLogger(__name__)

# Transient errors worth retrying
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def get_llm() -> BaseChatModel:
    return ChatOpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def ainvoke_with_retry(runnable, messages):
    """Call runnable.ainvoke(messages) with automatic retry on transient API errors.

    Retries up to 3 attempts with exponential backoff (2s, 4s, ..., max 30s)
    on rate-limit, timeout, connection, and server errors.
    """
    return await runnable.ainvoke(messages)
