import re

from ... import logger
from openai import AsyncOpenAI
from ...domain.entities import Question, RAGAnswer
from ...domain.repositories import ILLMProvider, LLMProviderError


class OpenAIClient(ILLMProvider):
    """OpenAI LLM provider implementation.

    This class implements the ILLMProvider interface using OpenAI's API
    to generate answers and analysis code based on user questions.
    """

    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key.
            model: OpenAI model to use for generation.
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    def _extract_answer_text(
        self,
        response_text: str
    ) -> str:
        """Extract and clean the natural language answer from the LLM response.

        Args:
            response_text: The raw response from the LLM.

        Returns:
            The cleaned answer text.
        """
        # Remove any code blocks and clean up the response
        lines = response_text.split('\n')
        answer_lines = []

        for line in lines:
            # Skip code blocks and empty lines
            if not line.strip().startswith('```') and line.strip():
                answer_lines.append(line.strip())

        # Join non-empty lines and return
        return ' '.join(answer_lines) if answer_lines else response_text.strip()

    async def generate_rag_answer(
        self,
        rag_prompt: str
    ) -> RAGAnswer:
        """Generate an answer using RAG context.

        Args:
            rag_prompt: The complete RAG prompt including context and question.

        Returns:
            A RAG answer based on retrieved documents.

        Raises:
            LLMProviderError: If the OpenAI API call fails.
        """
        try:
            # Load the rag answer prompt template from the configured file path
            with open('prompts/infrastructure/external/rag_answer_prompt.txt', 'r') as file:
                rag_system_prompt = file.read()

            # Call the OpenAI API with the RAG prompt
            # Use max_completion_tokens for newer models, fall back to max_tokens for older models
            completion_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": rag_system_prompt
                    },
                    {
                        "role": "user",
                        "content": rag_prompt
                    }
                ],
                "temperature": 0.1
            }

            # Add appropriate token limit parameter based on model
            if self.model.startswith("gpt-4o") or self.model.startswith("gpt-4-turbo") or self.model.startswith("o1"):
                completion_params["max_completion_tokens"] = 2000
            else:
                completion_params["max_tokens"] = 2000

            response = await self.client.chat.completions.create(**completion_params)
            response_text = response.choices[0].message.content

            # Extract the natural language answer
            answer_text = self._extract_answer_text(response_text)

            # Create and return a RAGAnswer (sources will be added by the RAG service)
            return RAGAnswer(
                text=answer_text,
                confidence_score=0.85,  # Good confidence for natural language responses
                sources=[],  # Sources will be populated by the RAG service
                explanation="Generated based on retrieved financial document context"
            )

        except FileNotFoundError:
            logger.error("RAG prompt file not found")
            raise
        except Exception as e:
            raise LLMProviderError(f"Failed to generate RAG answer: {str(e)}") from e
