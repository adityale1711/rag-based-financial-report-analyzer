import json
from ... import logger
from ...domain.entities import DatasetInfo, Question
from ...domain.repositories import IPromptService


class PromptService(IPromptService):
    """Implementation of the prompt service interface.

    This class builds comprehensive prompts for the LLM based on
    user questions and dataset information.
    """

    def __init__(self):
        """Initialize the PromptService."""
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM.

        Returns:
            A formatted system prompt string.
        """
        try:
            # Load the system prompt template from the configured file path
            with open('prompts/application/services/system_prompt.txt', 'r') as file:
                prompt = file.read()
            return prompt
        except FileNotFoundError:
            logger.error("Prompt file not found")
            raise

    def build_analysis_prompt(
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

        # Format sample data for better readability
        sample_data_str = json.dumps(dataset_info.sample_data[:5], indent=2)
        column_types_str = json.dumps(dataset_info.column_types, indent=2)
        available_columns = ", ".join(dataset_info.columns)

        try:
            # Load the prompt template from the configured file path
            with open('prompts/application/services/analysis_prompt.txt', 'r') as file:
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
