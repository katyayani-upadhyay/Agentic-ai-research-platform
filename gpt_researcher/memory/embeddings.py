"""Embedding provider management for GPT Researcher."""

import os
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings

OPENAI_EMBEDDING_MODEL = os.environ.get(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

_SUPPORTED_PROVIDERS = {
    "openai",
    "azure_openai",
    "cohere",
    "gigachat",
    "google_vertexai",
    "google_genai",
    "fireworks",
    "ollama",
    "together",
    "mistralai",
    "huggingface",
    "nomic",
    "voyageai",
    "dashscope",
    "custom",
    "bedrock",
    "aimlapi",
    "netmind",
    "openrouter",
    "minimax",
}


class Memory:
    def __init__(self, embedding_provider: str, model: str, **embedding_kwargs: Any):

        # FORCE FREE LOCAL EMBEDDINGS FOR ALL PROVIDERS
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self._embeddings = _embeddings

    def get_embeddings(self):
        return self._embeddings