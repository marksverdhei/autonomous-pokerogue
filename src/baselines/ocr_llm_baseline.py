"""
LLM Baseline that acts solely on OCR text input for game decisions.
"""

from typing import Dict


class OCRLLMBaseline:
    """
    LLM Baseline that makes decisions based purely on OCR text input.
    This baseline uses OCR extracted text to make game decisions.
    """

    def __init__(self, config: Dict):
        # Load LLM configuration
        self.llm_provider = config.get("llm_provider", "openrouter")
        self.model_name = config.get("model_name", "gpt-4-turbo")

    def get_action(self, observation: Dict, env=None) -> str:
        """
        Get action based on OCR text input using LLM.

        Args:
            observation: Current game state observation
            env: Game environment (optional)

        Returns:
            Action string for the game
        """
        # Extract OCR text from observation
        ocr_text = self._extract_ocr_text(observation)

        # Generate decision using LLM
        decision_prompt = self._create_decision_prompt(ocr_text, observation)
        action = self._generate_action_from_llm(decision_prompt)

        return action

    def _extract_ocr_text(self, observation: Dict) -> str:
        """
        Extract OCR text from observation.

        Args:
            observation: Current game state observation

        Returns:
            OCR text string
        """
        # Extract OCR data from the observation
        ocr_data = observation.get("ocr_data", [])

        # Format key game information for LLM
        formatted_ocr = "\n".join(ocr_data) if ocr_data else "No OCR data available"

        return formatted_ocr

    def _create_decision_prompt(self, ocr_text: str, observation: Dict) -> str:
        """
        Create a prompt for the LLM to make a decision.

        Args:
            ocr_text: OCR text extracted from game screen
            observation: Current game state

        Returns:
            Prompt string for the LLM
        """
        prompt = f"""
        You are an expert game player. Based on the following OCR text from a game,
        make a decision on what action to take.
        
        OCR Data:
        {ocr_text}
        
        Current Game State Context:
        - Available actions: {observation.get("available_actions", ["a", "b"])}
        
        Please respond with only the action name:
        - up
        - down
        - left
        - right
        - a
        - b
        - start
        
        What is your action?
        """
        return prompt.strip()

    def _generate_action_from_llm(self, prompt: str) -> str:
        """
        Generate action using LLM based on prompt.

        Args:
            prompt: Prompt to send to LLM

        Returns:
            Action string for the game
        """
        # This would normally call an LLM API
        # For now, we'll simulate with a simple rule-based approach
        # In a real implementation, you would use something like:
        #
        # response = openai.ChatCompletion.create(
        #     model=self.model_name,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.3
        # )
        # action_text = response.choices[0].message.content.strip()

        # Simulated LLM response - in practice, this would come from LLM API
        # We'll make a basic decision based on the OCR data

        # Simple heuristic for demonstration
        if "battle" in prompt.lower() or "fight" in prompt.lower():
            return "a"  # Attack
        elif "item" in prompt.lower():
            return "b"  # Use item
        elif "run" in prompt.lower():
            return "right"  # Run away
        else:
            return "a"  # Default to attack
