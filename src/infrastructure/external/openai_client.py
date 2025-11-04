import re
import json

from ... import logger
from openai import AsyncOpenAI
from ...domain.entities import AnalyzeCode, Answer, DatasetInfo, Question
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

    def _build_prompt(
        self, 
        question: Question, 
        dataset_info: DatasetInfo
    ) -> str:
        """Build a comprehensive prompt for the LLM.

        Args:
            question: The user's question.
            dataset_info: Information about the dataset.

        Returns:
            A formatted prompt string for the LLM.
        """

        sample_data_str = json.dumps(dataset_info.sample_data[:5], indent=2)
        column_types_str = json.dumps(dataset_info.column_types, indent=2)
        available_columns = ", ".join(dataset_info.columns)

        try:
            # Load the prompt template from the configured file path
            with open('prompts/infrastructure/external/builder_prompt.txt', 'r') as file:
                prompt_template = file.read()

            # Format the template with the specific arguments
            prompt = prompt_template.format(
                question=question, 
                dataset_info=dataset_info,
                available_columns=available_columns,
                sample_data_str=sample_data_str,
                column_types_str=column_types_str
            )

            return prompt
        except FileNotFoundError:
            logger.error("Prompt file not found")
            raise

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM.

        Returns:
            The system prompt string.
        """
        try:
            # Load the prompt template from the configured file path
            with open('prompts/infrastructure/external/system_prompt.txt', 'r') as file:
                prompt = file.read()
            return prompt
        except FileNotFoundError:
            logger.error("Prompt file not found")
            raise

    def _extract_text_answer(
        self,
        text: str
    ) -> str:
        """Extract the answer text from a non-JSON response.

        Args:
            text: The response text.

        Returns:
            The extracted answer text.
        """

        # Look for answer-like content before code blocks
        lines = text.split('\n')
        answer_lines = []

        for line in lines:
            if line.strip().startswith('```'):
                break
            if line.strip() and not line.lower().startswith({'answer:', 'question:'}):
                answer_lines.append(line.strip())

        return ' '.join(answer_lines) if answer_lines else text[:200]
    
    def _extract_code(
        self,
        text: str
    ) -> str:
        """Extract Python code from the response.

        Args:
            text: The response text.

        Returns:
            The extracted Python code.
        """

        # Look for code blocks
        code_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_pattern, text, re.DOTALL)

        if matches:
            return matches[0]
        
        # Alternative pattern for code without language specification
        alt_pattern = r'```\n(.*?)\n```'
        alt_pattern = re.findall(alt_pattern, text, re.DOTALL)

        if alt_pattern:
            return alt_pattern[0]
        
        # Fallback: look for code-like content
        return "# Could not extract code from response\nresult = None"

    def _parse_response(
        self,
        response_text: str
    ) -> tuple[Answer, AnalyzeCode]:
        """Parse the LLM response to extract answer and code.

        Args:
            response_text: The raw response from the LLM.

        Returns:
            A tuple containing the answer and analysis code.

        Raises:
            LLMProviderError: If the response cannot be parsed.
        """
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                response_data = json.loads(json_str)

                answer_text = response_data.get("answer", "No answer provided")
                confidence = float(response_data.get("confidence", 0.5))
                explanation = response_data.get("explanation", "")
                code = response_data.get("code", "# No code generated")
                code_description = response_data.get("code_description", "")

                answer = Answer(
                    text=answer_text,
                    confidence_score=confidence,
                    explanation=explanation
                )

                analysis_code = AnalyzeCode(
                    code=code,
                    description=code_description
                )

                return answer, analysis_code
            else:
                # Fallback: try to extract code and answer separately
                answer_text = self._extract_text_answer(response_text)
                code = self._extract_code(response_text)

                answer = Answer(
                    text=answer_text,
                    confidence_score=0.7,
                    explanation="Parsed from non-JSON response"
                )

                analysis_code = AnalyzeCode(
                    code=code,
                    description="Extracted from response"
                )

                return answer, analysis_code
        except Exception as e:
            raise LLMProviderError(f"Failed to generate answer: {str(e)}") from e

    async def generate_answer(
        self,
        question: Question,
        dataset_info: DatasetInfo
    ) -> tuple[Answer, AnalyzeCode]:
        """Generate an answer and analysis code using OpenAI API.

        Args:
            question: The user's question about the dataset.
            dataset_info: Information about the dataset structure.

        Returns:
            A tuple containing the answer and generated analysis code.

        Raises:
            LLMProviderError: If the OpenAI API call fails.
        """
        try:
            # Build the comprehensive prompt
            prompt = self._build_prompt(question, dataset_info)

            # Call the openAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            response_text = response.choices[0].message.content

            # Parse the response to extract answer and code
            answer, analyze_code = self._parse_response(response_text)

            return answer, analyze_code

        except Exception as e:
            raise LLMProviderError(f": {str(e)}") from e

    async def generate_rag_answer(
        self,
        rag_prompt: str
    ) -> tuple[Answer, AnalyzeCode]:
        """Generate an answer and analysis code using RAG context.

        Args:
            rag_prompt: The complete RAG prompt including context and question.

        Returns:
            A tuple containing the answer and generated analysis code.

        Raises:
            LLMProviderError: If the OpenAI API call fails.
        """
        try:
            # Load the rag answer prompt template from the configured file path
            with open('pprompts/infrastructure/external/rag_answer_prompt.txt', 'r') as file:
                rag_system_prompt = file.read()
            
            #  Call the OpenAI API with the RAG prompt
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": rag_system_prompt
                    },
                    {
                        "role": "user",
                        "content": rag_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            response_text = response.choices[0].message.content

            # Parse the response to extract answer and code
            answer, analyze_code = self._parse_response(response_text)

            return answer, analyze_code

        except FileNotFoundError:
            logger.error("Prompt file not found")
            raise

    async def generate_answer_with_prompt(
        self,
        question: Question,
        dataset_info: DatasetInfo,
        prompt: str
    ) -> tuple[Answer, AnalyzeCode]:
        """Generate an answer and analysis code using a pre-built prompt.

        Args:
            question: The user's question about the dataset.
            dataset_info: Information about the dataset structure.
            prompt: The complete prompt to send to the LLM.

        Returns:
            A tuple containing the answer and generated analysis code.

        Raises:
            LLMProviderError: If the OpenAI API call fails.
        """
        try:
            # Call the OpenAI API with the provided prompt
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            response_text = response.choices[0].message.content

            # Parse the response to extract answer and code
            answer, analyze_code = self._parse_response(response_text)

            return answer, analyze_code

        except Exception as e:
            raise LLMProviderError(f": {str(e)}") from e
