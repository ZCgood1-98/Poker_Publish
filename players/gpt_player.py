from .llm_base_player import LLMBasePlayer


class GPTPlayer(LLMBasePlayer):
    # Player alias for display purposes
    PLAYER_ALIAS = "Alex Chen"
    
    def get_model_config(self):
        """Return GPT model configuration"""
        return {
            'model': "gpt-4o-mini",
            'use_complex_messages': False
        }