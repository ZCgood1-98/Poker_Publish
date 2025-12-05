from .llm_base_player import LLMBasePlayer


class QwenPlayer(LLMBasePlayer):
    # Player alias for display purposes
    PLAYER_ALIAS = "Robert Garcia"
    
    def get_model_config(self):
        """Return Qwen 2.5-7B model configuration"""
        return {
            'model': "qwen/qwen-2.5-7b-instruct",
            'use_complex_messages': True
        } 