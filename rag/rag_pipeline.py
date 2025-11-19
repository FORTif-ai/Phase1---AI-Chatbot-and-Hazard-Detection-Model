"""
RAG Pipeline for Fortif.ai.
Implements Retrieval, Augmentation, and Generation steps.
"""

import logging
from typing import List, Tuple, Optional

import weaviate
from weaviate.classes.query import Filter, MetadataQuery
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import settings
from prompts import (
    get_rag_prompt_template,
    get_empty_context_prompt_template,
    contains_medical_query,
    get_medical_advice_deflection
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for Fortif.ai chatbot.

    Handles:
    1. Retrieval: Vector similarity search with filters
    2. Augmentation: Context formatting from retrieved documents
    3. Generation: LLM-based response generation
    """

    def __init__(
        self,
        weaviate_client: weaviate.WeaviateClient,
        embeddings: GoogleGenerativeAIEmbeddings,
        llm: ChatGoogleGenerativeAI
    ):
        """
        Initialize RAG pipeline.

        Args:
            weaviate_client: Connected Weaviate client
            embeddings: Google embeddings model
            llm: Google LLM for generation
        """
        self.client = weaviate_client
        self.embeddings = embeddings
        self.llm = llm

        # Initialize prompt templates
        self.rag_prompt = get_rag_prompt_template()
        self.empty_context_prompt = get_empty_context_prompt_template()

        # Create LangChain chains
        self.rag_chain = self.rag_prompt | llm | StrOutputParser()
        self.empty_context_chain = self.empty_context_prompt | llm | StrOutputParser()

        logger.info("RAG pipeline initialized successfully")

    def retrieve(
        self,
        patient_id: str,
        question: str,
        limit: int = 3,
        include_sensitive: bool = False,
        emotion_filter: Optional[str] = None
    ) -> List:
        """
        Retrieval step: Search for relevant patient memories.

        Args:
            patient_id: Patient identifier
            question: User's question
            limit: Maximum number of documents to retrieve
            include_sensitive: Whether to include sensitive memories
            emotion_filter: Optional emotion filter (positive/negative/neutral/mixed/unknown)

        Returns:
            List of retrieved document objects from Weaviate
        """
        try:
            # Generate query embedding
            logger.info(f"Generating embedding for query: '{question[:50]}...'")
            query_vector = self.embeddings.embed_query(
                question,
                output_dimensionality=settings.vector_dimension
            )

            # Build filter
            logger.info(f"Building filters for patient_id={patient_id}")
            query_filter = Filter.by_property("patient_id").equal(patient_id)

            if not include_sensitive:
                query_filter = query_filter & Filter.by_property("is_sensitive").equal(False)
                logger.info("Excluding sensitive memories")

            if emotion_filter:
                query_filter = query_filter & Filter.by_property("emotion").equal(emotion_filter)
                logger.info(f"Filtering by emotion: {emotion_filter}")

            # Perform hybrid search (Keyword + Vector)
            logger.info(f"Searching Weaviate (Hybrid) for {limit} relevant memories...")
            collection = self.client.collections.get(settings.weaviate_collection_name)
            response = collection.query.hybrid(
                query=question,
                vector=query_vector,
                limit=limit,
                filters=query_filter,
                alpha=0.5,  # Balanced between keyword (0.0) and vector (1.0)
                return_metadata=MetadataQuery(score=True, explain_score=True)
            )

            logger.info(f"Retrieved {len(response.objects)} documents")
            return response.objects

        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            raise

    def augment(self, retrieved_docs: List) -> Tuple[str, bool]:
        """
        Augmentation step: Format retrieved documents into context string.

        Args:
            retrieved_docs: List of retrieved Weaviate objects

        Returns:
            Tuple of (formatted_context_string, has_context)
        """
        if not retrieved_docs:
            logger.info("No documents retrieved - empty context")
            return "", False

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            props = doc.properties

            # Format each memory with metadata for richer context
            memory_text = (
                f"Memory {i} (Topic: {props['topic']}, Emotion: {props['emotion']}, "
                f"Source: {props['source']}):\n{props['text']}"
            )
            context_parts.append(memory_text)

        context = "\n\n".join(context_parts)
        logger.info(f"Formatted context from {len(retrieved_docs)} documents ({len(context)} chars)")

        # Truncate if exceeds max length
        if len(context) > settings.max_context_length:
            context = context[:settings.max_context_length] + "..."
            logger.warning(f"Context truncated to {settings.max_context_length} characters")

        return context, True

    def generate(self, context: str, question: str, has_context: bool) -> str:
        """
        Generation step: Use LLM to generate empathetic response.

        Args:
            context: Formatted context from retrieved documents
            question: User's question
            has_context: Whether context contains relevant information

        Returns:
            Generated response string
        """
        try:
            # Safety check for medical advice requests
            if contains_medical_query(question):
                logger.warning(f"Medical advice query detected: {question[:50]}...")
                return get_medical_advice_deflection()

            # Choose appropriate chain based on context availability
            if has_context:
                logger.info("Generating response with context")
                response = self.rag_chain.invoke({
                    "context": context,
                    "question": question
                })
            else:
                logger.info("Generating response without context (inviting sharing)")
                response = self.empty_context_chain.invoke({
                    "question": question
                })

            logger.info(f"Generated response ({len(response)} chars)")
            return response.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            # Fallback to safe default response
            return (
                "I'm having a bit of trouble right now, but I'm here to listen. "
                "Could you tell me more about what's on your mind?"
            )

    def run(
        self,
        patient_id: str,
        question: str,
        limit: int = 3,
        include_sensitive: bool = False,
        emotion_filter: Optional[str] = None
    ) -> Tuple[str, List]:
        """
        Execute full RAG pipeline: Retrieve -> Augment -> Generate.

        Args:
            patient_id: Patient identifier
            question: User's question
            limit: Maximum documents to retrieve
            include_sensitive: Include sensitive memories
            emotion_filter: Optional emotion filter

        Returns:
            Tuple of (generated_response, retrieved_documents)
        """
        logger.info(f"Starting RAG pipeline for patient_id={patient_id}")

        # Step 1: Retrieval
        docs = self.retrieve(
            patient_id=patient_id,
            question=question,
            limit=limit,
            include_sensitive=include_sensitive,
            emotion_filter=emotion_filter
        )

        # Step 2: Augmentation
        context, has_context = self.augment(docs)

        # Step 3: Generation
        response = self.generate(context, question, has_context)

        logger.info("RAG pipeline completed successfully")
        return response, docs
