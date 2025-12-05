from .llm_base_player import LLMBasePlayer


class ClaudePlayer(LLMBasePlayer):
    # Player alias for display purposes
    PLAYER_ALIAS = "Michael Davis"
    
    def get_model_config(self):
        """Return Claude model configuration"""
        return {
            'model': "anthropic/claude-3.5-haiku",
            'use_complex_messages': True
        }