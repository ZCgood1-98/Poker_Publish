from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.engine.card import Card
from openai import OpenAI
import json
import re
import os
from abc import ABC, abstractmethod


class LLMBasePlayer(BasePokerPlayer, ABC):
    """
    Base LLM poker player class containing all common functionality.
    Subclasses only need to implement get_model_config method to specify the model to use.
    """
    
    def __init__(self):
        super().__init__()
        # Initialize tracking variables
        self.initial_stack = 0
        self.current_stack = 0
        self.total_gains_losses = 0
        self.round_gains_losses = {}
        self.parse_failed = 0
        self.current_street = None
        self.current_hand_strength = 0 
        self.current_hand_type = None
        # Store reasoning process for each round
        self.reasoning_history = {}  # {round_count: {street: {'reasoning': str, 'action': dict}}}
        self.current_round_count = 0
        
        # Track players and action history
        self.player_names = {}  # {uuid: name}
        self.current_round_actions = {}  # {street: [{'player': name, 'action': str, 'amount': int}]}
        self.seats_info = {}  # {uuid: {'name': str, 'stack': int, 'position': int}}
        
        self.hand_stats = {
            'played': 0,
            'won': 0,
            'folded': 0,
            'called': 0,
            'raised': 0
        }
        self.actions_by_street = {
            'preflop': {'fold': 0, 'call': 0, 'raise': 0, 'profit': 0},
            'flop': {'fold': 0, 'call': 0, 'raise': 0, 'profit': 0},
            'turn': {'fold': 0, 'call': 0, 'raise': 0, 'profit': 0},
            'river': {'fold': 0, 'call': 0, 'raise': 0, 'profit': 0}
        }
        self.value_bets = {'made': 0, 'successful': 0, 'profit': 0}
        self.bluffs = {'made': 0, 'successful': 0, 'profit': 0}
        self.aggression_actions = {'raises': 0, 'calls': 0}
        self.hand_strength_actions = {
            'very_weak': {'fold': 0, 'call': 0, 'raise': 0},    # 0.0-0.25
            'weak': {'fold': 0, 'call': 0, 'raise': 0},         # 0.25-0.5
            'medium': {'fold': 0, 'call': 0, 'raise': 0},       # 0.5-0.75
            'strong': {'fold': 0, 'call': 0, 'raise': 0}        # 0.75-1.0
        }
        self.hand_strength_by_street = {
            'preflop': [],
            'flop': [],
            'turn': [],
            'river': []
        }    
        self.street_investment = {
            'preflop': 0,
            'flop': 0, 
            'turn': 0,
            'river': 0
        }

    @abstractmethod
    def get_model_config(self):
        """
        Subclasses must implement this method to return model configuration dictionary
        
        Returns:
            dict: Dictionary containing the following keys:
                - model: model name (str)
                - use_complex_messages: whether to use complex message format (bool)
                - extra_prompt: additional prompt text (str, optional)
        """
        pass

    def declare_action(self, valid_actions, hole_card, round_state):
        fold = valid_actions[0]
        call = valid_actions[1]
        raise_action = valid_actions[2]
        
        community_card = round_state['community_card']
        pot_size = round_state['pot']['main']['amount']
        street = round_state['street']
        
        current_highest_bet = 0
        my_current_bet = 0
        
        if street in round_state['action_histories']:
            for action_history in round_state['action_histories'][street]:
                if 'amount' in action_history:
                    current_highest_bet = max(current_highest_bet, action_history['amount'])
                
                if action_history['uuid'] == self.uuid and 'amount' in action_history:
                    my_current_bet = action_history['amount']
        
        call_amount = current_highest_bet - my_current_bet
        pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0
        
        min_raise = raise_action['amount']['min']
        max_raise = raise_action['amount']['max']
        
        # Calculate hand strength
        if not community_card:
            rank1, rank2 = hole_card[0][1], hole_card[1][1]
            rank_values = {"A": 14, "K": 13, "Q": 12, "J": 11, "T": 10}
            
            # Handle rank conversion more safely
            try:
                r1 = rank_values.get(rank1, int(rank1))
                r2 = rank_values.get(rank2, int(rank2))
            except ValueError:
                # Fallback for invalid ranks
                r1 = rank_values.get(rank1, 7)
                r2 = rank_values.get(rank2, 7)
            suited = hole_card[0][0] == hole_card[1][0]
            
            if r1 == r2:
                self.current_hand_strength = 0.5 + (r1 / 28)
                self.current_hand_type = "PAIR"
            elif suited:
                self.current_hand_strength = 0.3 + ((r1 + r2) / 60)
                self.current_hand_type = "SUITED_CARDS"
            else:
                self.current_hand_strength = 0.2 + ((r1 + r2) / 60)
                self.current_hand_type = "OFFSUIT_CARDS"
        else:
            try:
                parsed_hole = [Card.from_str(c) for c in hole_card]
                parsed_community = [Card.from_str(c) for c in community_card]
                hand_info = HandEvaluator.gen_hand_rank_info(parsed_hole, parsed_community)
                if hand_info and 'hand' in hand_info and 'strength' in hand_info['hand']:
                    self.current_hand_type = hand_info['hand']['strength']
                else:
                    self.current_hand_type = "UNKNOWN"
                
                score = HandEvaluator.eval_hand(parsed_hole, parsed_community)
                
                # Normalize the score (HandEvaluator returns scores from 0 to 7462,
                # with 7462 being the highest)
                max_score = 7462
                self.current_hand_strength = score / max_score
            except Exception as e:
                print(f"Hand evaluation error: {e}")
                # Fallback to simple calculation
                self.current_hand_type = "UNKNOWN"
                self.current_hand_strength = 0.5  # Default middle strength
        
        if street in self.hand_strength_by_street:
            self.hand_strength_by_street[street].append(self.current_hand_strength)
        
        # Get position and table information
        position_info = self.get_position_info(round_state)
        
        # Get current round's action history
        opponent_actions = self.get_formatted_opponent_actions(round_state)
        
        game_state = {
            "hole_cards": hole_card,
            "community_cards": community_card,
            "pot_size": pot_size,
            "street": street,
            "call_amount": call_amount,
            "min_raise": min_raise,
            "max_raise": max_raise,
            "pot_odds": pot_odds,
            "hand_type": self.current_hand_type,
            "hand_strength": self.current_hand_strength,
            "opponent_actions": opponent_actions,  # Added for context
            "position_info": position_info  # Added position and stack info
        }
        
        # Call LLM API and get reasoning and decision
        action_decision, reasoning = self.call_llm_api(game_state)
        
        # Save reasoning history
        if self.current_round_count not in self.reasoning_history:
            self.reasoning_history[self.current_round_count] = {}
        self.reasoning_history[self.current_round_count][street] = {
            'reasoning': reasoning,
            'action': action_decision,
            'game_state': game_state.copy()
        }
        
        # 显示推理过程
        model_name = self.get_model_config()['model'].split('/')[-1]
        alias = getattr(self.__class__, 'PLAYER_ALIAS', 'Unknown Player')
        print(f"\n🤖 [{model_name} ({alias})] Round {self.current_round_count} - {street.upper()}")
        print(f"💭 Reasoning: {reasoning}")
        print(f"🎯 Decision: {action_decision}")
        
        # Update statistics
        if street in self.actions_by_street:
            action_type = action_decision["action"].lower()
            if action_type in self.actions_by_street[street]:
                self.actions_by_street[street][action_type] += 1
        
        # Update All of the tracking Stats
        if action_decision["action"] == "RAISE":
            self.aggression_actions['raises'] += 1
        elif action_decision["action"] == "CALL":
            self.aggression_actions['calls'] += 1
            
        if self.current_hand_strength < 0.25:
            category = 'very_weak'
        elif self.current_hand_strength < 0.5:
            category = 'weak'
        elif self.current_hand_strength < 0.75:
            category = 'medium'
        else:
            category = 'strong'
        
        if action_decision["action"] in self.hand_strength_actions[category]:
            self.hand_strength_actions[category][action_decision["action"].lower()] += 1
            
        self.hand_stats['played'] += 1
        
        if action_decision["action"] == "FOLD":
            self.hand_stats['folded'] += 1
            # return fold['action'], fold['amount']
            return fold['action'], fold['amount']
        elif action_decision["action"] == "CALL":
            self.hand_stats['called'] += 1
            return call['action'], call['amount']
        elif action_decision["action"] == "RAISE":
            self.hand_stats['raised'] += 1
            
            # Track value betting and bluffing attempts
            if self.current_hand_strength >= 0.7:
                self.value_bets['made'] += 1
            else:
                self.bluffs['made'] += 1
            
            # Ensure raise amount is within valid range
            if 'amount' in action_decision:
                intended_amount = int(action_decision['amount'])
                actual_amount = max(min_raise, min(intended_amount, max_raise))
                return raise_action['action'], actual_amount
            else:
                return raise_action['action'], min_raise
        else:
            return fold['action'], fold['amount']

    def call_llm_api(self, game_state):
        config = self.get_model_config()
        if not config:
            print("No model config found. Defaulting to FOLD.")
            return {"action": "FOLD"}, "No model configuration"
        
        # Format opponent action history
        opponent_actions_text = self.format_opponent_actions_for_prompt(game_state.get('opponent_actions', {}))
        
        # Format position and table information
        position_info = game_state.get('position_info', {})
        position_text = self.format_position_info_for_prompt(position_info)
        
        base_prompt = f"""
You are a professional poker player, playing against other professional poker players, and you aim to make the most money in the long run. 

Current game state:
- Your hole cards: {game_state['hole_cards']}
- Community cards: {game_state['community_cards']}
- Current street: {game_state['street']}
- Hand type: {game_state['hand_type']}
- Hand strength: {game_state['hand_strength']:.4f}
- Pot size: {game_state['pot_size']}
- Call amount: {game_state['call_amount']}
- Minimum raise: {game_state['min_raise']}
- Maximum raise: {game_state['max_raise']}
- Pot odds: {game_state['pot_odds']:.2f}

{position_text}

{opponent_actions_text}

Please provide your analysis and decision in the following format:

ANALYSIS:
[Provide a concise analysis of the current situation, considering hand strength, pot odds, position, opponent actions, and optimal strategy]

DECISION:
{{"action": "FOLD/CALL/RAISE", "amount": integer_if_raise}}

Important:
- Keep your analysis concise but insightful (2-3 sentences)
- Consider pot odds, hand strength, opponent behavior patterns, and strategic implications
- Pay attention to opponent aggression levels and betting patterns for potential bluffs or value bets
- The JSON must be valid and parsable
- Only include "amount" field if action is "RAISE"

Examples:
ANALYSIS: Strong pocket pair with good pot odds. Opponent has been passive, so betting for value against likely weaker hands.
DECISION: {{"action": "RAISE", "amount": 50}}

ANALYSIS: Weak hand with poor pot odds and aggressive opponent raising. Folding to preserve chips against likely stronger range.
DECISION: {{"action": "FOLD"}}"""
        
        # Add extra prompt if specified
        extra_prompt = config.get('extra_prompt')
        if extra_prompt:
            base_prompt += f"\n{extra_prompt}"

        # Get API key from environment variable
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("Warning: OPENROUTER_API_KEY environment variable not set. Using default fallback.")
            api_key = "your-api-key-here"  # Placeholder - will fail gracefully
            
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Use different message formats based on model config
        use_complex_messages = config.get('use_complex_messages', True)
        if use_complex_messages:
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": base_prompt
                        }
                    ]
                }
            ]
        else:
            messages = [{"role": "user", "content": base_prompt}]

        model_name = config.get('model', 'unknown-model')
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3
        )

        llm_response = response.choices[0].message.content
        # print(f"[{model_name}] Raw LLM response: {llm_response}")  # 注释掉，避免输出混乱

        # 解析推理和决策
        reasoning, decision = self.parse_llm_response(llm_response)
        
        return decision, reasoning

    def get_formatted_opponent_actions(self, round_state):
        """
        从round_state中提取并格式化对手的操作历史
        """
        formatted_actions = {}
        
        # 更新座位信息
        for seat in round_state.get('seats', []):
            uuid = seat.get('uuid')
            if uuid:
                self.seats_info[uuid] = {
                    'name': seat.get('name', f'Player_{uuid[:8]}'),
                    'stack': seat.get('stack', 0),
                    'state': seat.get('state', 'unknown')
                }
                self.player_names[uuid] = seat.get('name', f'Player_{uuid[:8]}')
        
        # 遍历每个街道的操作历史
        action_histories = round_state.get('action_histories', {})
        
        for street_name, actions in action_histories.items():
            if street_name not in formatted_actions:
                formatted_actions[street_name] = []
            
            for action in actions:
                uuid = action.get('uuid')
                if uuid and uuid != self.uuid:  # 不包括自己的操作
                    player_name = self.player_names.get(uuid, f'Player_{uuid[:8]}')
                    action_type = action.get('action', 'UNKNOWN')
                    amount = action.get('amount', 0)
                    
                    # 格式化操作描述
                    if action_type == 'FOLD':
                        action_desc = "FOLD"
                    elif action_type == 'CALL':
                        action_desc = f"CALL {amount}" if amount > 0 else "CALL"
                    elif action_type == 'RAISE':
                        action_desc = f"RAISE to {amount}"
                    elif action_type == 'SMALLBLIND':
                        action_desc = f"SMALL BLIND {amount}"
                    elif action_type == 'BIGBLIND':
                        action_desc = f"BIG BLIND {amount}"
                    else:
                        action_desc = f"{action_type} {amount}" if amount > 0 else action_type
                    
                    formatted_actions[street_name].append({
                        'player': player_name,
                        'action': action_desc,
                        'stack': self.seats_info.get(uuid, {}).get('stack', 0)
                    })
        
        return formatted_actions

    def format_opponent_actions_for_prompt(self, opponent_actions):
        """
        将对手操作历史格式化为prompt中的文本
        """
        if not opponent_actions:
            return "Opponent Actions: No actions yet this round."
        
        actions_text = "Opponent Actions This Round:\n"
        
        for street, actions in opponent_actions.items():
            if actions:  # 只显示有操作的街道
                actions_text += f"- {street.capitalize()}:\n"
                for action_info in actions:
                    player = action_info['player']
                    action = action_info['action']
                    stack = action_info['stack']
                    actions_text += f"  * {player} (Stack: {stack}): {action}\n"
        
        # 如果没有任何非盲注操作，显示默认信息
        if actions_text == "Opponent Actions This Round:\n":
            actions_text = "Opponent Actions: Only blinds posted so far."
        
        return actions_text

    def format_position_info_for_prompt(self, position_info):
        """
        将位置和桌面信息格式化为prompt中的简洁文本
        """
        if not position_info:
            return "Table Info: Position information unavailable."
        
        total_players = position_info.get('total_players', 0)
        my_position = position_info.get('my_position', 'Unknown')
        my_stack = position_info.get('my_stack', 0)
        players_info = position_info.get('players_info', [])
        
        # 简洁的桌面信息
        table_text = f"Table Info:\n"
        table_text += f"- Players: {total_players}, Your position: {my_position}, Your stack: {my_stack}\n"
        
        # 只显示对手的筹码信息（不显示自己）
        opponents = [p for p in players_info if not p.get('is_me', False)]
        if opponents:
            table_text += f"- Opponent stacks:"
            for player in opponents:
                table_text += f" {player['name']}({player['stack']})"
        
        return table_text

    def parse_llm_response(self, llm_response):
        """解析LLM回复，提取推理和JSON决策"""
        try:
            # 提取分析部分
            reasoning = ""
            if "ANALYSIS:" in llm_response:
                analysis_start = llm_response.find("ANALYSIS:") + len("ANALYSIS:")
                if "DECISION:" in llm_response:
                    analysis_end = llm_response.find("DECISION:")
                    reasoning = llm_response[analysis_start:analysis_end].strip()
                else:
                    # 如果没有找到DECISION标记，尝试提取到JSON之前的内容
                    json_start = llm_response.find("{")
                    if json_start > analysis_start:
                        reasoning = llm_response[analysis_start:json_start].strip()
                    else:
                        reasoning = llm_response[analysis_start:].strip()
            
            # 提取JSON决策
            json_match = re.search(r'\{[^}]*"action"[^}]*\}', llm_response)
            if json_match:
                json_str = json_match.group()
                decision = json.loads(json_str)
            else:
                # 尝试提取 DECISION: 后面的JSON
                if "DECISION:" in llm_response:
                    decision_start = llm_response.find("DECISION:") + len("DECISION:")
                    decision_text = llm_response[decision_start:].strip()
                    
                    # 提取JSON部分（可能包含在代码块中）
                    if decision_text.startswith("```"):
                        decision_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", decision_text.strip())
                    
                    # 查找JSON对象
                    json_match = re.search(r'\{.*?\}', decision_text, re.DOTALL)
                    if json_match:
                        decision = json.loads(json_match.group())
                    else:
                        raise ValueError("No valid JSON found in decision section")
                else:
                    raise ValueError("No DECISION section found")
            
            # 如果没有提取到推理，使用默认值
            if not reasoning:
                reasoning = "No detailed analysis provided"
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse LLM response: {e}")
            self.parse_failed += 1
            reasoning = f"Parse failed: {str(e)}"
            decision = {"action": "FOLD"}
            
        return reasoning, decision
              

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']
        # Game info gives :
        # player_num (Number of players playing)
        # rules : {initial_stack': 100, 'max_round': 1, 'small_blind_amount': 5, 'ante': 0, 'blind_structure': {}}
        # seats (Players playing information) : {'name': 'Fish', 'uuid': 'ujjwyahvpvpeigcatpcxmh', 'stack': 100, 'state': 'participating'}
                
        # Initialize starting stack
        for player in game_info['seats']:
            if player['uuid'] == self.uuid:
                self.initial_stack = player['stack']
                self.current_stack = player['stack']
                break
            
        # Reset tracking for new game
        self.total_gains_losses = 0
        self.round_gains_losses = {}
        self.reasoning_history = {}

    def receive_round_start_message(self, round_count, hole_card, seats):
        # Save current round count
        self.current_round_count = round_count
        
        # 重置当前轮次的操作历史
        self.current_round_actions = {
            'preflop': [],
            'flop': [],
            'turn': [],
            'river': []
        }
        
        # 1 round is the entire process of Flop, Turn, and River
        # round_count (counts the round it is current in)
        # hole_card (holds the hole card information of the player)
        # seats (Players playing information) (stack = money)
        for seat in seats:
            if seat['uuid'] == self.uuid:
                stack_before_round = self.current_stack
                self.current_stack = seat['stack']
                
                # If it's not the first round, calculate gain/loss from previous round
                if round_count > 1:
                    previous_round = round_count - 1
                    round_result = self.current_stack - stack_before_round
                    self.round_gains_losses[previous_round] = round_result
                    self.total_gains_losses = self.current_stack - self.initial_stack
                break
        pass

    def receive_street_start_message(self, street, round_state):
        # street is preflop, flop, turn, river
        self.current_street = street
        # round_state gives : 
        # street : {'street' : 'preflop'}
        # pot : {'main': {'amount': 15}, 'side': []}
        # 'community_card': [], 'dealer_btn': 0, 'next_player': 0, 'small_blind_pos': 1, 'big_blind_pos': 2, 'round_count': 1, 'small_blind_amount': 5
        # seats (Players playing information) (stack = money)
        # action histories : {'preflop': [{'action': 'SMALLBLIND', 'amount': 5, 'add_amount': 5, 'uuid': 'aqkqtioikejuttreankpsh'}
        for seat in round_state['seats']:
            if seat['uuid'] == self.uuid:
                self.current_stack = seat['stack']
                break
        pass

    def receive_game_update_message(self, action, round_state):
        # action returns the actions of every player on the table
        # round_state gives : 
        # the street : {'street' : 'preflop'}
        # the pot : {'main': {'amount': 15}, 'side': []}
        # 'community_card': [], 'dealer_btn': 0, 'next_player': 0, 'small_blind_pos': 1, 'big_blind_pos': 2, 'round_count': 1, 'small_blind_amount': 5
        # seats (Players playing information) (stack = money)
        # action histories : {'preflop': [{'action': 'SMALLBLIND', 'amount': 5, 'add_amount': 5, 'uuid': 'aqkqtioikejuttreankpsh'}
        
        for seat in round_state['seats']:
            if seat['uuid'] == self.uuid:
                self.current_stack = seat['stack']
                break
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # Keep existing functionality
        round_count = round_state['round_count']
        
        # Check if I won this hand
        for winner in winners:
            if winner['uuid'] == self.uuid:
                self.hand_stats['won'] += 1
                break
        
        # Find my final stack for this round
        for seat in round_state['seats']:
            if seat['uuid'] == self.uuid:
                new_stack = seat['stack']
                round_result = new_stack - self.current_stack
                self.current_stack = new_stack
                self.round_gains_losses[round_count] = round_result
                self.total_gains_losses = self.current_stack - self.initial_stack
                
                # Update street profit
                if self.current_street:
                    self.actions_by_street[self.current_street]['profit'] += round_result
                
                # Update value bet or bluff success
                went_to_showdown = len(hand_info) > 0
                if went_to_showdown:
                    if self.current_hand_strength >= 0.7:  # Value bet threshold
                        self.value_bets['successful'] += 1 if round_result > 0 else 0
                        self.value_bets['profit'] += round_result
                    else:
                        self.bluffs['successful'] += 1 if round_result > 0 else 0
                        self.bluffs['profit'] += round_result
                break
        pass
    
    def get_performance_stats(self):
        """Return comprehensive performance statistics"""
        hands_played = self.hand_stats['played']
        win_rate = (self.hand_stats['won'] / hands_played) * 100 if hands_played > 0 else 0
        fold_rate = (self.hand_stats['folded'] / hands_played) * 100 if hands_played > 0 else 0
        call_rate = (self.hand_stats['called'] / hands_played) * 100 if hands_played > 0 else 0
        raise_rate = (self.hand_stats['raised'] / hands_played) * 100 if hands_played > 0 else 0
                
        # Aggression Factor
        af_calls = self.aggression_actions['calls'] or 1  # Avoid division by zero
        aggression_factor = self.aggression_actions['raises'] / af_calls
        
        # Value bet and bluff success rates
        value_bet_success_rate = (self.value_bets['successful'] / self.value_bets['made']) * 100 if self.value_bets['made'] > 0 else 0
        bluff_success_rate = (self.bluffs['successful'] / self.bluffs['made']) * 100 if self.bluffs['made'] > 0 else 0
        
        return {
            'initial_stack': self.initial_stack,
            'current_stack': self.current_stack,
            'total_profit_loss': self.total_gains_losses,
            'profit_percentage': (self.total_gains_losses / self.initial_stack) * 100 if self.initial_stack > 0 else 0,
            'round_results': self.round_gains_losses,
            'hands_played': hands_played,
            'hands_won': self.hand_stats['won'],
            'hands_folded': self.hand_stats['folded'],
            'hands_called': self.hand_stats['called'],
            'hands_raised': self.hand_stats['raised'],
            'win_rate': win_rate,
            'fold_rate': fold_rate,
            'call_rate': call_rate,
            'raise_rate': raise_rate,
            'aggression_factor': aggression_factor,
            'actions_by_street': self.actions_by_street,
            'value_betting': {
                'attempts': self.value_bets['made'],
                'successful': self.value_bets['successful'],
                'success_rate': value_bet_success_rate,
                'profit': self.value_bets['profit']
            },
            'bluffing': {
                'attempts': self.bluffs['made'],
                'successful': self.bluffs['successful'],
                'success_rate': bluff_success_rate,
                'profit': self.bluffs['profit']
            },
            'hand_strength_decisions': self.hand_strength_actions,
            'failed_parses': self.parse_failed,
            'reasoning_history': self.reasoning_history  # 新增：推理历史
        }

    def get_reasoning_summary(self):
        """获取推理过程的摘要"""
        model_name = self.get_model_config()['model'].split('/')[-1]
        summary = f"\n📊 {model_name} Reasoning Summary:\n"
        summary += "=" * 50 + "\n"
        
        for round_num, round_data in self.reasoning_history.items():
            summary += f"\n🎲 Round {round_num}:\n"
            for street, street_data in round_data.items():
                summary += f"  📈 {street.capitalize()}:\n"
                summary += f"    💭 {street_data['reasoning']}\n"
                summary += f"    🎯 {street_data['action']}\n"
        
        return summary 

    def get_position_info(self, round_state):
        """
        获取位置信息、玩家数量、玩家顺序和筹码信息
        """
        # 获取基本信息
        seats = round_state.get('seats', [])
        total_players = len([seat for seat in seats if seat.get('state') == 'participating'])
        dealer_btn = round_state.get('dealer_btn', 0)
        
        # 获取自己的位置信息
        my_position = None
        my_stack = 0
        
        # 创建按位置排序的玩家列表
        players_by_position = []
        
        for i, seat in enumerate(seats):
            if seat.get('state') == 'participating':
                player_info = {
                    'name': seat.get('name', f'Player_{seat.get("uuid", "")[:8]}'),
                    'stack': seat.get('stack', 0),
                    'position': i,
                    'is_dealer': i == dealer_btn,
                    'is_me': seat.get('uuid') == self.uuid
                }
                
                if player_info['is_me']:
                    my_position = i
                    my_stack = player_info['stack']
                
                players_by_position.append(player_info)
        
        # 计算相对位置描述
        position_desc = "Unknown"
        if my_position is not None and total_players > 1:
            if total_players == 2:
                # 头对头
                position_desc = "Dealer/SB" if my_position == dealer_btn else "BB"
            else:
                # 多人桌
                positions_from_dealer = (my_position - dealer_btn) % total_players
                if positions_from_dealer == 0:
                    position_desc = "Dealer"
                elif positions_from_dealer == 1:
                    position_desc = "Small Blind"
                elif positions_from_dealer == 2:
                    position_desc = "Big Blind"
                elif positions_from_dealer <= total_players // 2:
                    position_desc = "Early Position"
                elif positions_from_dealer <= 3 * total_players // 4:
                    position_desc = "Middle Position"
                else:
                    position_desc = "Late Position"
        
        return {
            'total_players': total_players,
            'my_position': position_desc,
            'my_stack': my_stack,
            'players_info': players_by_position
        } 