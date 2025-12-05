from .llm_base_player import LLMBasePlayer
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.engine.card import Card
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import random

NB_SIMULATION = 1000


class MontePlayer(LLMBasePlayer):
    # Player alias for display purposes
    PLAYER_ALIAS = "David Wilson"
    
    def get_model_config(self):
        """Return Monte Carlo algorithm configuration"""
        return {
            'model': "monte-carlo-algorithm",
            'use_complex_messages': False
        }
    
    def call_llm_api(self, game_state):
        """Override LLM API call to use Monte Carlo algorithm instead"""
        hole_card = game_state['hole_cards']
        community_card = game_state['community_cards']
        pot_size = game_state['pot_size']
        street = game_state['street']
        call_amount = game_state['call_amount']
        min_raise = game_state['min_raise']
        max_raise = game_state['max_raise']
        
        # Calculate win rate using Monte Carlo simulation
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=NB_SIMULATION,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card)
        )

        # Calculate pot odds
        pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0

        # Decision logic based on Monte Carlo simulation
        action_decision = None
        reasoning = ""
        
        if win_rate > 0.8:
            action_decision = {"action": "RAISE", "amount": max_raise}
            reasoning = f"Monte Carlo simulation shows very high win rate ({win_rate:.2%}). Maximum value bet is optimal."

        elif win_rate > 0.6:
            bet_amount = int(0.75 * pot_size)
            bet_amount = min(max(bet_amount, min_raise), max_raise)
            action_decision = {"action": "RAISE", "amount": bet_amount}
            reasoning = f"Strong win rate ({win_rate:.2%}) indicates profitable raise. Betting 75% of pot size for value."

        elif win_rate > 0.4:
            action_decision = {"action": "RAISE", "amount": min_raise}
            reasoning = f"Moderate win rate ({win_rate:.2%}) suggests minimum raise for value/protection."

        elif win_rate < 0.3 and pot_odds < 0.2:
            action_decision = {"action": "FOLD", "amount": 0}
            reasoning = f"Low win rate ({win_rate:.2%}) and poor pot odds ({pot_odds:.2%}) make folding optimal."

        else:
            action_decision = {"action": "CALL", "amount": call_amount}
            reasoning = f"Win rate ({win_rate:.2%}) and pot odds ({pot_odds:.2%}) justify calling to see next card."

        return action_decision, reasoning

    def receive_game_start_message(self, game_info):
        """Override to store nb_player for Monte Carlo simulation"""
        super().receive_game_start_message(game_info)
        self.nb_player = game_info['player_num']