#!/usr/bin/env python3
"""
Poker Game Visualization Tool - Web interface for viewing specific rounds and reasoning
"""

import json
import os
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import webbrowser
import threading
import time

class PokerVisualizationHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, experiment_data=None, **kwargs):
        self.experiment_data = experiment_data
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.serve_main_page()
        elif self.path.startswith('/api/'):
            self.handle_api_request()
        else:
            self.send_error(404)
    
    def serve_main_page(self):
        html_content = self.generate_html()
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def handle_api_request(self):
        try:
            if self.path == '/api/matches':
                self.serve_matches_data()
            elif self.path.startswith('/api/round/'):
                self.serve_round_data()
            else:
                self.send_error(404)
        except Exception as e:
            self.send_error(500, str(e))
    
    def serve_matches_data(self):
        if not self.experiment_data:
            self.send_error(500, "No experiment data loaded")
            return
        
        matches_info = []
        for i, battle in enumerate(self.experiment_data['battles']):
            player_names = list(battle['players'].keys())
            matches_info.append({
                'match_id': i,
                'players': player_names,
                'rounds': self.get_available_rounds(battle)
            })
        
        response = json.dumps(matches_info)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))
    
    def get_available_rounds(self, battle):
        rounds = set()
        for player_info in battle['players'].values():
            reasoning_history = player_info.get('reasoning_history', {})
            for round_num in reasoning_history.keys():
                if round_num.isdigit():
                    rounds.add(int(round_num))
        return sorted(list(rounds))
    
    def serve_round_data(self):
        # Parse URL: /api/round/{match_id}/{round_num}
        path_parts = self.path.split('/')
        if len(path_parts) < 5:
            self.send_error(400, "Invalid round request format")
            return
        
        try:
            match_id = int(path_parts[3])
            round_num = int(path_parts[4])
        except ValueError:
            self.send_error(400, "Invalid match_id or round_num")
            return
        
        if not self.experiment_data or match_id >= len(self.experiment_data['battles']):
            self.send_error(404, "Match not found")
            return
        
        battle = self.experiment_data['battles'][match_id]
        round_data = self.extract_round_data(battle, round_num)
        
        if not round_data:
            self.send_error(404, "Round not found")
            return
        
        response = json.dumps(round_data)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))
    
    def extract_round_data(self, battle, round_num):
        round_str = str(round_num)
        round_data = {
            'round_num': round_num,
            'streets': {},
            'players': {}
        }
        
        # Extract data for each player
        for player_name, player_info in battle['players'].items():
            reasoning_history = player_info.get('reasoning_history', {})
            if round_str not in reasoning_history:
                continue
            
            player_round_data = reasoning_history[round_str]
            round_data['players'][player_name] = {
                'initial_stack': player_info.get('initial_stack', 1000),
                'streets': {}
            }
            
            # Extract data for each street
            for street in ['preflop', 'flop', 'turn', 'river']:
                if street in player_round_data:
                    street_data = player_round_data[street]
                    if isinstance(street_data, dict):
                        game_state = street_data.get('game_state', {})
                        action = street_data.get('action', {})
                        reasoning = street_data.get('reasoning', '')
                        
                        player_street_data = {
                            'hole_cards': game_state.get('hole_cards', []),
                            'community_cards': game_state.get('community_cards', []),
                            'pot_size': game_state.get('pot_size', 0),
                            'my_stack': game_state.get('position_info', {}).get('my_stack', 0),
                            'action': action,
                            'reasoning': reasoning
                        }
                        
                        round_data['players'][player_name]['streets'][street] = player_street_data
                        
                        # Store community cards at street level for easy access
                        if street not in round_data['streets']:
                            round_data['streets'][street] = {
                                'community_cards': game_state.get('community_cards', []),
                                'pot_size': game_state.get('pot_size', 0)
                            }
        
        return round_data if round_data['players'] else None
    
    def generate_html(self):
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PokerBench Result Visualizer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f4c3a 0%, #1a5f4a 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1800px;
            margin: 0 auto;
            padding: 8px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .controls {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }
        
        .control-group {
            display: inline-block;
            margin: 0 15px 10px 0;
        }
        
        .control-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        select, button {
            padding: 10px 15px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        select {
            background: white;
            color: #333;
            min-width: 150px;
        }
        
        button {
            background: #e74c3c;
            color: white;
            font-weight: bold;
        }
        
        button:hover {
            background: #c0392b;
            transform: translateY(-2px);
        }
        
        button.active {
            background: #27ae60;
        }
        
        .streets-container {
            display: flex;
            gap: 8px;
            margin: 15px 0;
            overflow-x: auto;
        }
        
        .street-panel {
            flex: 1;
            min-width: 320px;
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 12px;
            backdrop-filter: blur(10px);
        }
        
        .street-header {
            text-align: center;
            margin-bottom: 15px;
            border-bottom: 2px solid rgba(255,255,255,0.2);
            padding-bottom: 10px;
        }
        
        .street-header h3 {
            margin-bottom: 8px;
            font-size: 1.2em;
            color: #f39c12;
        }
        
        .community-cards-small {
            display: flex;
            justify-content: center;
            gap: 3px;
            margin-bottom: 5px;
        }
        
        .card-small {
            width: 32px;
            height: 44px;
            background: white;
            border-radius: 4px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            font-weight: bold;
            font-size: 12px;
        }
        
        .card-small.red {
            color: #e74c3c;
        }
        
        .card-small.black {
            color: #2c3e50;
        }
        
        .card-small-back {
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
        }
        
        .pot-info-small {
            font-size: 0.9em;
            color: #27ae60;
            font-weight: bold;
        }
        
        .players-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .player-compact {
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
            display: flex;
            gap: 6px;
        }
        
        .player-left {
            flex: 0 0 100px;
            display: flex;
            flex-direction: column;
        }
        
        .player-right {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .player-name-compact {
            font-weight: bold;
            color: #f39c12;
            margin-bottom: 6px;
            font-size: 1.0em;
        }
        
        .player-hole-cards {
            display: flex;
            gap: 3px;
            margin-bottom: 6px;
        }
        
        .player-info-compact {
            display: flex;
            flex-direction: column;
            gap: 3px;
            font-size: 0.9em;
        }
        
        .player-action {
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.85em;
        }
        
        .action-fold {
            background-color: #e74c3c;
            color: white;
        }
        
        .action-call {
            background-color: #f39c12;
            color: white;
        }
        
        .action-raise {
            background-color: #27ae60;
            color: white;
        }
        
        .action-check {
            background-color: #3498db;
            color: white;
        }
        
        .action-allin {
            background-color: #9b59b6;
            color: white;
        }
        
        .action-default {
            background-color: #95a5a6;
            color: white;
        }
        
        .player-stack {
            color: #3498db;
        }
        
        .reasoning-compact {
            background: rgba(0,0,0,0.2);
            padding: 8px;
            border-radius: 4px;
            font-size: 0.85em;
            line-height: 1.4;
            height: 100%;
            overflow: visible;
        }
        
        .loading {
            text-align: center;
            padding: 50px;
            font-size: 1.2em;
        }
        
        .error {
            background: rgba(231, 76, 60, 0.2);
            border: 2px solid #e74c3c;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .controls {
                padding: 15px;
            }
            
            .control-group {
                display: block;
                margin-bottom: 15px;
            }
            
            .players-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🃏 PokerBench Result Visualizer</h1>
            <p>Explore detailed poker rounds and player reasoning</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="matchSelect">Match:</label>
                <select id="matchSelect">
                    <option value="">Select a match...</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="roundSelect">Round:</label>
                <select id="roundSelect" disabled>
                    <option value="">Select a round...</option>
                </select>
            </div>
            
            <div class="control-group">
                <button id="loadRound" disabled>Load Round</button>
            </div>
        </div>
        
        <div id="gameContent" style="display: none;">
            <div class="streets-container">
                <div class="street-panel" data-street="preflop">
                    <div class="street-header">
                        <h3>Pre-flop</h3>
                        <div class="community-cards-small" id="preflop-community"></div>
                        <div class="pot-info-small" id="preflop-pot"></div>
                    </div>
                    <div class="players-list" id="preflop-players"></div>
                </div>
                
                <div class="street-panel" data-street="flop">
                    <div class="street-header">
                        <h3>Flop</h3>
                        <div class="community-cards-small" id="flop-community"></div>
                        <div class="pot-info-small" id="flop-pot"></div>
                    </div>
                    <div class="players-list" id="flop-players"></div>
                </div>
                
                <div class="street-panel" data-street="turn">
                    <div class="street-header">
                        <h3>Turn</h3>
                        <div class="community-cards-small" id="turn-community"></div>
                        <div class="pot-info-small" id="turn-pot"></div>
                    </div>
                    <div class="players-list" id="turn-players"></div>
                </div>
                
                <div class="street-panel" data-street="river">
                    <div class="street-header">
                        <h3>River</h3>
                        <div class="community-cards-small" id="river-community"></div>
                        <div class="pot-info-small" id="river-pot"></div>
                    </div>
                    <div class="players-list" id="river-players"></div>
                </div>
            </div>
        </div>
        
        <div id="loading" class="loading" style="display: none;">
            Loading...
        </div>
        
        <div id="error" class="error" style="display: none;">
            <!-- Error messages will be shown here -->
        </div>
    </div>

    <script>
        let currentRoundData = null;
        
        // Card suit symbols
        const SUITS = {
            'H': '♥',
            'D': '♦',
            'C': '♣',
            'S': '♠'
        };
        
        const SUIT_COLORS = {
            'H': 'red',
            'D': 'red',
            'C': 'black',
            'S': 'black'
        };
        
        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            loadMatches();
            setupEventListeners();
        });
        
        function setupEventListeners() {
            document.getElementById('matchSelect').addEventListener('change', onMatchChange);
            document.getElementById('loadRound').addEventListener('click', loadRound);
        }
        
        async function loadMatches() {
            try {
                showLoading(true);
                const response = await fetch('/api/matches');
                if (!response.ok) throw new Error('Failed to load matches');
                
                const matches = await response.json();
                populateMatchSelect(matches);
                showLoading(false);
            } catch (error) {
                showError('Failed to load matches: ' + error.message);
                showLoading(false);
            }
        }
        
        function populateMatchSelect(matches) {
            const select = document.getElementById('matchSelect');
            select.innerHTML = '<option value="">Select a match...</option>';
            
            matches.forEach(match => {
                const option = document.createElement('option');
                option.value = match.match_id;
                option.textContent = `Match ${match.match_id + 1} (${match.players.join(', ')})`;
                option.dataset.rounds = JSON.stringify(match.rounds);
                select.appendChild(option);
            });
        }
        
        function onMatchChange() {
            const select = document.getElementById('matchSelect');
            const roundSelect = document.getElementById('roundSelect');
            const loadButton = document.getElementById('loadRound');
            
            if (select.value) {
                const rounds = JSON.parse(select.selectedOptions[0].dataset.rounds);
                populateRoundSelect(rounds);
                roundSelect.disabled = false;
            } else {
                roundSelect.innerHTML = '<option value="">Select a round...</option>';
                roundSelect.disabled = true;
                loadButton.disabled = true;
            }
        }
        
        function populateRoundSelect(rounds) {
            const select = document.getElementById('roundSelect');
            const loadButton = document.getElementById('loadRound');
            
            select.innerHTML = '<option value="">Select a round...</option>';
            rounds.forEach(round => {
                const option = document.createElement('option');
                option.value = round;
                option.textContent = `Round ${round}`;
                select.appendChild(option);
            });
            
            select.addEventListener('change', function() {
                loadButton.disabled = !this.value;
            });
        }
        
        async function loadRound() {
            const matchId = document.getElementById('matchSelect').value;
            const roundNum = document.getElementById('roundSelect').value;
            
            if (!matchId || !roundNum) return;
            
            try {
                showLoading(true);
                const response = await fetch(`/api/round/${matchId}/${roundNum}`);
                if (!response.ok) throw new Error('Failed to load round data');
                
                currentRoundData = await response.json();
                currentStreet = 'preflop';
                displayRound();
                showLoading(false);
            } catch (error) {
                showError('Failed to load round: ' + error.message);
                showLoading(false);
            }
        }
        
        function displayRound() {
            if (!currentRoundData) return;
            
            document.getElementById('gameContent').style.display = 'block';
            displayAllStreets();
        }
        
        function displayAllStreets() {
            const streets = ['preflop', 'flop', 'turn', 'river'];
            
            streets.forEach(street => {
                displayStreetCommunityCards(street);
                displayStreetPlayers(street);
            });
        }
        
        function displayStreetCommunityCards(street) {
            const communityContainer = document.getElementById(`${street}-community`);
            const potContainer = document.getElementById(`${street}-pot`);
            
            if (currentRoundData.streets[street]) {
                const streetData = currentRoundData.streets[street];
                const cards = streetData.community_cards || [];
                
                communityContainer.innerHTML = '';
                cards.forEach(card => {
                    communityContainer.appendChild(createSmallCardElement(card));
                });
                
                // Always show 5 community card positions
                const totalCards = 5;
                
                // For preflop, show all cards as face-down
                if (street === 'preflop') {
                    for (let i = 0; i < 5; i++) {
                        communityContainer.appendChild(createSmallCardElement(null, true));
                    }
                } else {
                    // For other streets, show revealed cards + face-down placeholders for remaining
                    for (let i = cards.length; i < totalCards; i++) {
                        communityContainer.appendChild(createSmallCardElement(null, true));
                    }
                }
                
                potContainer.innerHTML = `Pot: ${streetData.pot_size || 0}`;
            } else {
                communityContainer.innerHTML = '<span style="font-size: 0.8em;">No data</span>';
                potContainer.innerHTML = '';
            }
        }
        
        function displayStreetPlayers(street) {
            const container = document.getElementById(`${street}-players`);
            container.innerHTML = '';
            
            Object.entries(currentRoundData.players).forEach(([playerName, playerData]) => {
                const streetData = playerData.streets[street];
                if (streetData) {
                    container.appendChild(createCompactPlayerCard(playerName, streetData, playerData.initial_stack));
                }
            });
        }
        
        // Model name mapping from alias to real model name
        const MODEL_NAMES = {
            'Alex Chen': 'GPT-4o mini',
            'Sarah Johnson': 'Llama 4 Maverick', 
            'Michael Davis': 'Claude 3.5 Haiku',
            'Emily Zhang': 'DeepSeek Chat V3',
            'Robert Garcia': 'Qwen 2.5 Coder',
            'Jessica Liu': 'Gemini 2.5 Flash',
            'David Wilson': 'Monte Carlo'
        };
        
        function getModelName(aliasName) {
            return MODEL_NAMES[aliasName] || aliasName;
        }
        
        function getActionClass(action) {
            if (!action || !action.action) return 'action-default';
            
            const actionType = action.action.toLowerCase();
            switch(actionType) {
                case 'fold': return 'action-fold';
                case 'call': return 'action-call';
                case 'raise': return 'action-raise';
                case 'check': return 'action-check';
                case 'allin': return 'action-allin';
                default: return 'action-default';
            }
        }
        
        function createCompactPlayerCard(playerName, streetData, initialStack) {
            const card = document.createElement('div');
            card.className = 'player-compact';
            
            const modelName = getModelName(playerName);
            
            const holeCardsHtml = streetData.hole_cards.map(cardStr => 
                createSmallCardElement(cardStr).outerHTML
            ).join('');
            
            const actionText = streetData.action ? 
                `${streetData.action.action}${streetData.action.amount ? ' (' + streetData.action.amount + ')' : ''}` : 
                'No action';
            
            const actionClass = getActionClass(streetData.action);
            
            card.innerHTML = `
                <div class="player-left">
                    <div class="player-name-compact">${modelName}</div>
                    <div class="player-hole-cards">${holeCardsHtml}</div>
                    <div class="player-info-compact">
                        <span class="player-action ${actionClass}">${actionText}</span>
                        <span class="player-stack">Stack: ${streetData.my_stack || initialStack}</span>
                    </div>
                </div>
                <div class="player-right">
                    <div class="reasoning-compact">${streetData.reasoning || 'No reasoning available'}</div>
                </div>
            `;
            
            return card;
        }
        
        function createSmallCardElement(cardStr, isBack = false) {
            const card = document.createElement('div');
            card.className = 'card-small';
            
            if (isBack) {
                card.classList.add('card-small-back');
                card.innerHTML = '?';
                return card;
            }
            
            if (!cardStr || cardStr.length < 2) {
                card.innerHTML = '?';
                return card;
            }
            
            const suit = cardStr[0];
            const rank = cardStr.slice(1);
            const displayRank = rank === 'T' ? '10' : rank;
            const suitSymbol = SUITS[suit] || suit;
            const color = SUIT_COLORS[suit] || 'black';
            
            card.classList.add(color);
            card.innerHTML = `
                <div style="font-size: ${displayRank === '10' ? '8px' : '10px'};">${displayRank}</div>
                <div style="font-size: 12px;">${suitSymbol}</div>
            `;
            
            return card;
        }
        
        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.innerHTML = message;
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }
    </script>
</body>
</html>'''

def load_experiment_data(experiment_path: str):
    """Load experiment data from results.json"""
    results_file = os.path.join(experiment_path, 'results.json')
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Experiment results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)

def list_experiments():
    """List all available experiments"""
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        print("Experiments directory not found")
        return []
    
    experiments = [d for d in os.listdir(experiments_dir) 
                  if os.path.isdir(os.path.join(experiments_dir, d))]
    
    if not experiments:
        print("No experiments found")
        return []
    
    print("Available experiments:")
    for i, exp in enumerate(sorted(experiments)):
        print(f"  {i+1}. {exp}")
    
    return sorted(experiments)

def create_handler_class(experiment_data):
    """Create a handler class with experiment data"""
    class CustomHandler(PokerVisualizationHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, experiment_data=experiment_data, **kwargs)
    return CustomHandler

def start_server(experiment_data, port=8080):
    """Start the HTTP server"""
    handler_class = create_handler_class(experiment_data)
    server = HTTPServer(('localhost', port), handler_class)
    
    print(f"Starting poker visualization server on http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    # Open browser automatically
    def open_browser():
        time.sleep(1)  # Wait for server to start
        webbrowser.open(f'http://localhost:{port}')
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()

def main():
    parser = argparse.ArgumentParser(description='Poker Game Visualization Tool')
    parser.add_argument('--experiment', type=str, help='Name of experiment to visualize')
    parser.add_argument('--port', type=int, default=8080, help='Port to run server on (default: 8080)')
    parser.add_argument('--list', action='store_true', help='List all available experiments')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    # If no experiment specified, let user choose
    if not args.experiment:
        experiments = list_experiments()
        if not experiments:
            return
        
        try:
            choice = int(input("\nEnter experiment number: ")) - 1
            if 0 <= choice < len(experiments):
                args.experiment = experiments[choice]
            else:
                print("Invalid choice")
                return
        except (ValueError, KeyboardInterrupt):
            print("Cancelled")
            return
    
    experiment_path = os.path.join("experiments", args.experiment)
    
    if not os.path.exists(experiment_path):
        print(f"Experiment does not exist: {args.experiment}")
        return
    
    try:
        print(f"Loading experiment: {args.experiment}")
        experiment_data = load_experiment_data(experiment_path)
        print(f"Loaded {len(experiment_data['battles'])} matches")
        
        start_server(experiment_data, args.port)
        
    except Exception as e:
        print(f"Failed to start visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
