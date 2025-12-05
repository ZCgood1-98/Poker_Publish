from .llm_base_player import LLMBasePlayer


class GeminiPlayer(LLMBasePlayer):
    # Player alias for display purposes
    PLAYER_ALIAS = "Jessica Liu"
    
    def get_model_config(self):
        """Return Gemini model configuration"""
        return {
            'model': "google/gemini-2.5-flash-preview-09-2025",
            'use_complex_messages': True,
            'extra_prompt': 'You are an analytical poker player. Make decisions based on mathematical probability and opponent behavior patterns.'
        } 