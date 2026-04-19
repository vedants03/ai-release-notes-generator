from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
import os

load_dotenv()


def get_llm() -> BaseChatModel:
    return ChatOpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
    )
