from .llm_base_player import LLMBasePlayer


class DeepSeekPlayer(LLMBasePlayer):
    # Player alias for display purposes
    PLAYER_ALIAS = "Emily Zhang"
    
    def get_model_config(self):
        """Return DeepSeek model configuration"""
        return {
            'model': 'deepseek/deepseek-chat-v3-0324',
            'use_complex_messages': False,
            'extra_prompt': 'You are a conservative poker player. Focus on value betting and avoid unnecessary risks.'
        } 