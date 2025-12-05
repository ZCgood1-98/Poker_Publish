#!/usr/bin/env python3
"""
Poker Game Evaluation Tool - Analyze experiment results and generate academic tables and figures
"""

import json
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，适用于WSL环境
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict

# Set font and academic style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5

def load_experiment_data(experiment_path: str) -> Dict:
    """Load experiment data"""
    results_file = os.path.join(experiment_path, 'results.json')
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Experiment results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)

def get_player_display_name(player_id: str) -> str:
    """Get player display name - use model name directly"""
    # Map from alias to model name
    alias_to_model = {
        'Alex Chen': 'GPT-4o mini',
        'Sarah Johnson': 'Llama 4 Maverick', 
        'Michael Davis': 'Claude 3.5 Haiku',
        'David Wilson': 'Monte Carlo',
        'Emily Zhang': 'DeepSeek Chat V3',
        'Robert Garcia': 'Qwen 2.5 Instruct',
        'Jessica Liu': 'Gemini 2.5 Flash'
    }
    
    # If it's an alias, convert to model name
    if player_id in alias_to_model:
        return alias_to_model[player_id]
    
    # If it's a model ID, map directly
    model_mapping = {
        'gpt': 'GPT-4o mini',
        'llama': 'Llama 4 Maverick',
        'claude': 'Claude 3.5 Haiku',
        'monte': 'Monte Carlo',
        'deepseek': 'DeepSeek Chat V3',
        'qwen': 'Qwen 2.5 Instruct',
        'gemini': 'Gemini 2.5 Flash'
    }
    
    return model_mapping.get(player_id, player_id)

def calculate_metrics(data: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Calculate statistics for three metrics"""
    battles = data['battles']
    
    # Collect data for all players
    player_data = {}
    
    for battle in battles:
        for player_name, player_info in battle['players'].items():
            if player_name not in player_data:
                player_data[player_name] = {
                    'accumulated_profits': [],
                    'avg_chips': [],
                    'total_hands_won': 0,
                    'total_hands_played': 0
                }
            
            # 1. Accumulated profit (percentage relative to initial chips)
            initial_stack = player_info['initial_stack']
            total_profit_loss = player_info['total_profit_loss']
            accumulated_profit = (total_profit_loss / initial_stack) * 100
            player_data[player_name]['accumulated_profits'].append(accumulated_profit)
            
            # 2. Accumulate hands data for calculating overall win rate
            hands_won = player_info.get('hands_won', 0)
            hands_played = player_info.get('hands_played', 0)
            player_data[player_name]['total_hands_won'] += hands_won
            player_data[player_name]['total_hands_played'] += hands_played
            
            # 3. Average chips (chips at the end of each match)
            current_stack = player_info['current_stack']
            player_data[player_name]['avg_chips'].append(current_stack)
    
    # Calculate statistics
    results = {}
    for player_name, data_lists in player_data.items():
        results[player_name] = {}
        
        # 1. Accumulated profit statistics
        accumulated_profits = data_lists['accumulated_profits']
        if accumulated_profits:
            results[player_name]['accumulated_profit'] = {
                'mean': np.mean(accumulated_profits),
                'std': np.std(accumulated_profits, ddof=1) if len(accumulated_profits) > 1 else 0.0
            }
        else:
            results[player_name]['accumulated_profit'] = {'mean': 0.0, 'std': 0.0}
        
        # 2. Overall win rate calculation (correct method)
        total_hands_won = data_lists['total_hands_won']
        total_hands_played = data_lists['total_hands_played']
        overall_win_rate = (total_hands_won / total_hands_played * 100) if total_hands_played > 0 else 0.0
        results[player_name]['win_rate'] = {
            'mean': overall_win_rate,
            'std': 0.0  # Overall win rate is a single value, no standard deviation
        }
        
        # 3. Average chips statistics
        avg_chips_list = data_lists['avg_chips']
        if avg_chips_list:
            results[player_name]['avg_chips'] = {
                'mean': np.mean(avg_chips_list),
                'std': np.std(avg_chips_list, ddof=1) if len(avg_chips_list) > 1 else 0.0
            }
        else:
            results[player_name]['avg_chips'] = {'mean': 0.0, 'std': 0.0}
    
    return results

def calculate_behavior_stats(data: Dict) -> Dict[str, Dict[str, float]]:
    """Calculate behavioral statistics for each agent"""
    battles = data['battles']
    
    # Collect behavioral data for all players
    player_behavior_data = {}
    
    for battle in battles:
        for player_name, player_info in battle['players'].items():
            if player_name not in player_behavior_data:
                player_behavior_data[player_name] = {
                    'total_hands_played': 0,
                    'total_hands_folded': 0,
                    'total_hands_called': 0,
                    'total_hands_raised': 0,
                    'total_aggression_factor_sum': 0.0,
                    'battle_count': 0
                }
            
            # Accumulate data
            player_behavior_data[player_name]['total_hands_played'] += player_info.get('hands_played', 0)
            player_behavior_data[player_name]['total_hands_folded'] += player_info.get('hands_folded', 0)
            player_behavior_data[player_name]['total_hands_called'] += player_info.get('hands_called', 0)
            player_behavior_data[player_name]['total_hands_raised'] += player_info.get('hands_raised', 0)
            player_behavior_data[player_name]['total_aggression_factor_sum'] += player_info.get('aggression_factor', 0.0)
            player_behavior_data[player_name]['battle_count'] += 1
    
    # Calculate statistical results
    behavior_stats = {}
    for player_name, data in player_behavior_data.items():
        total_hands = data['total_hands_played']
        if total_hands > 0:
            fold_rate = (data['total_hands_folded'] / total_hands) * 100
            call_rate = (data['total_hands_called'] / total_hands) * 100
            raise_rate = (data['total_hands_raised'] / total_hands) * 100
        else:
            fold_rate = call_rate = raise_rate = 0.0
        
        # Calculate average aggression factor
        avg_aggression_factor = data['total_aggression_factor_sum'] / data['battle_count'] if data['battle_count'] > 0 else 0.0
        
        behavior_stats[player_name] = {
            'hands_played': total_hands,
            'fold_rate': fold_rate,
            'call_rate': call_rate,
            'raise_rate': raise_rate,
            'aggression_factor': avg_aggression_factor
        }
    
    return behavior_stats

def collect_hand_strength_actions(data: Dict) -> Dict[str, List[Tuple[float, str]]]:
    """Collect hand strength and corresponding action data for each agent"""
    battles = data['battles']
    
    # Collect hand strength-action data for each player
    player_hand_strength_actions = defaultdict(list)
    
    for battle in battles:
        for player_name, player_info in battle['players'].items():
            reasoning_history = player_info.get('reasoning_history', {})
            
            # Iterate through reasoning history for each round
            for round_num, round_data in reasoning_history.items():
                # Check data for each street
                for street in ['preflop', 'flop', 'turn', 'river']:
                    if street in round_data and isinstance(round_data[street], dict):
                        street_data = round_data[street]
                        
                        # Extract hand strength and action
                        if ('game_state' in street_data and 
                            'hand_strength' in street_data['game_state'] and
                            'action' in street_data):
                            
                            hand_strength = street_data['game_state']['hand_strength']
                            action_data = street_data['action']
                            
                            if isinstance(action_data, dict) and 'action' in action_data:
                                action = action_data['action']
                                
                                # Add hand strength and action to list
                                player_hand_strength_actions[player_name].append((hand_strength, action))
    
    return dict(player_hand_strength_actions)

def print_results_table(results: Dict[str, Dict[str, Dict[str, float]]]):
    """Print results table"""
    print("\nTable 1. Main Quantitative results of PokerBench")
    print("=" * 80)
    print(f"{'Agent':<20} {'Accumulated Profit (%)':<25} {'Per-Round Win Rate (%)':<22} {'Avg. Chips/Match':<15}")
    print("-" * 80)
    
    # Sort by player names for consistency
    sorted_players = sorted(results.keys())
    
    for player in sorted_players:
        display_name = get_player_display_name(player)
        metrics = results[player]
        
        # Format values as mean±std
        profit_mean = metrics['accumulated_profit']['mean']
        profit_std = metrics['accumulated_profit']['std']
        winrate_mean = metrics['win_rate']['mean']
        winrate_std = metrics['win_rate']['std']
        chips_mean = metrics['avg_chips']['mean']
        chips_std = metrics['avg_chips']['std']
        
        profit_str = f"{profit_mean:.1f}±{profit_std:.1f}"
        winrate_str = f"{winrate_mean:.1f}"  # Win Rate doesn't show std, as it's an overall statistic
        chips_str = f"{chips_mean:.1f}±{chips_std:.1f}"
        
        print(f"{display_name:<20} {profit_str:<25} {winrate_str:<22} {chips_str:<15}")
    
    print("=" * 80)

def print_behavior_stats_table(behavior_stats: Dict[str, Dict[str, float]]):
    """Print behavioral statistics table"""
    print("\nTable 2. Behavioral Statistics of PokerBench Agents")
    print("=" * 95)
    print(f"{'Agent':<20} {'Hands Played':<12} {'Fold Rate (%)':<12} {'Call Rate (%)':<12} {'Raise Rate (%)':<13} {'Aggression Factor':<15}")
    print("-" * 95)
    
    # Sort by player names for consistency
    sorted_players = sorted(behavior_stats.keys())
    
    for player in sorted_players:
        display_name = get_player_display_name(player)
        stats = behavior_stats[player]
        
        hands_played = int(stats['hands_played'])
        fold_rate = stats['fold_rate']
        call_rate = stats['call_rate']
        raise_rate = stats['raise_rate']
        aggression_factor = stats['aggression_factor']
        
        print(f"{display_name:<20} {hands_played:<12} {fold_rate:<12.1f} {call_rate:<12.1f} {raise_rate:<13.1f} {aggression_factor:<15.2f}")
    
    print("=" * 95)

def generate_chips_over_rounds_figure(data: Dict, experiment_name: str):
    """Generate average chips over rounds figure"""
    battles = data['battles']
    
    # Collect chip data for each player in each battle for each round
    player_battle_chips = {}
    
    for battle_idx, battle in enumerate(battles):
        for player_name, player_info in battle['players'].items():
            if player_name not in player_battle_chips:
                player_battle_chips[player_name] = []
            
            # Create chip array for each round, extract real chips from game state
            initial_stack = player_info['initial_stack']
            battle_chips = [initial_stack]  # Round 0 = initial chips
            
            # Extract chip data for each round (1-50) from reasoning_history
            reasoning_history = player_info.get('reasoning_history', {})
            
            for round_num in range(1, 51):
                round_key = str(round_num)
                round_chips = initial_stack  # Default value
                
                # Find reasoning history data for this round
                if round_key in reasoning_history:
                    round_data = reasoning_history[round_key]
                    
                    # Use preflop chips as chips at the beginning of this round
                    if 'preflop' in round_data and isinstance(round_data['preflop'], dict):
                        preflop_data = round_data['preflop']
                        if 'game_state' in preflop_data:
                            game_state = preflop_data['game_state']
                            if 'position_info' in game_state and 'my_stack' in game_state['position_info']:
                                round_chips = game_state['position_info']['my_stack']
                
                battle_chips.append(round_chips)
            
            player_battle_chips[player_name].append(battle_chips)
    
    # Calculate average chips and standard deviation for each player per round
    player_avg_chips = {}
    player_std_chips = {}
    
    for player_name, battle_list in player_battle_chips.items():
        if battle_list:
            # Convert to numpy array and calculate mean and standard deviation
            chips_array = np.array(battle_list)
            avg_chips = np.mean(chips_array, axis=0).tolist()
            std_chips = np.std(chips_array, axis=0).tolist()
            player_avg_chips[player_name] = avg_chips
            player_std_chips[player_name] = std_chips
    
    # Generate figure - adjust to longer rectangle suitable for single-column papers
    plt.figure(figsize=(14, 6))
    
    # Define color scheme (colorblind-friendly)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Round range (0-50)
    max_rounds = 50
    rounds = list(range(max_rounds + 1))
    
    for i, (player_name, avg_chips) in enumerate(player_avg_chips.items()):
        display_name = get_player_display_name(player_name)
        std_chips = player_std_chips[player_name]
        color = colors[i % len(colors)]
        
        # Calculate upper and lower bounds
        upper_bound = [avg + std for avg, std in zip(avg_chips, std_chips)]
        lower_bound = [avg - std for avg, std in zip(avg_chips, std_chips)]
        
        # Plot average line
        plt.plot(rounds, avg_chips, 
                label=display_name, 
                color=color,
                linewidth=2,
                marker='o' if i < 3 else 's',  # Different marker styles
                markersize=3,
                markevery=5)  # Show marker every 5 points
        
        # Add standard deviation shaded area
        plt.fill_between(rounds, lower_bound, upper_bound, 
                        color=color, alpha=0.2)
        
        # Find minimum and maximum points in standard deviation range
        min_idx = np.argmin(lower_bound)
        max_idx = np.argmax(upper_bound)
        min_value = lower_bound[min_idx]
        max_value = upper_bound[max_idx]
        min_round = rounds[min_idx]
        max_round = rounds[max_idx]
        
        # Plot minimum and maximum points
        plt.scatter(min_round, min_value, color=color, s=50, marker='v', zorder=5)  # Down triangle for minimum
        plt.scatter(max_round, max_value, color=color, s=50, marker='^', zorder=5)  # Up triangle for maximum
        
        # Add value annotations
        plt.annotate(f'{min_value:.0f}', 
                    xy=(min_round, min_value), 
                    xytext=(5, -15), 
                    textcoords='offset points',
                    fontsize=9, 
                    color=color,
                    fontweight='bold',
                    ha='left')
        
        plt.annotate(f'{max_value:.0f}', 
                    xy=(max_round, max_value), 
                    xytext=(5, 10), 
                    textcoords='offset points',
                    fontsize=9, 
                    color=color,
                    fontweight='bold',
                    ha='left')
    
    plt.xlabel('Round', fontsize=14, fontweight='bold')
    plt.ylabel('Average Chips', fontsize=14, fontweight='bold')
    # Remove title, will be added separately in paper
    plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)  # Set X-axis range, remove white borders
    plt.tight_layout()
    
    # Save as PDF format
    filename = f"{experiment_name}_chips_over_rounds.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
    plt.close()
    
    print(f"\nFigure saved: {filename}")

def generate_hand_strength_action_figure(hand_strength_data: Dict[str, List[Tuple[float, str]]], experiment_name: str):
    """Generate hand strength-action distribution figure"""
    
    # Filter out players with no data
    filtered_data = {player: data for player, data in hand_strength_data.items() if data}
    
    if not filtered_data:
        print("No hand strength data found, skipping hand strength-action distribution figure")
        return
    
    # Calculate subplot layout: 4 in first row, 3 in second row
    n_players = len(filtered_data)
    if n_players <= 4:
        rows, cols = 1, n_players
    else:
        rows, cols = 2, 4
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    
    # Convert axes to unified indexing method
    if n_players == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Define action colors
    action_colors = {
        'FOLD': '#d62728',      # Red
        'CALL': '#2ca02c',      # Green  
        'RAISE': '#1f77b4',     # Blue
        'CHECK': '#ff7f0e',     # Orange
        'ALL_IN': '#9467bd'     # Purple
    }
    
    # Define hand strength intervals - adjusted for better distribution
    strength_bins = [0, 0.5, 2, 10, 50, float('inf')]
    strength_labels = ['Very Weak\n(0-0.5)', 'Weak\n(0.5-2)', 'Medium\n(2-10)', 'Strong\n(10-50)', 'Very Strong\n(50+)']
    
    player_names = sorted(filtered_data.keys())
    
    for idx, player_name in enumerate(player_names):
        ax = axes_flat[idx]
        
        data = filtered_data[player_name]
        display_name = get_player_display_name(player_name)
        
        # Group data by hand strength intervals and actions
        strength_action_counts = defaultdict(lambda: defaultdict(int))
        
        for hand_strength, action in data:
            # Determine hand strength category
            strength_category = None
            for i, threshold in enumerate(strength_bins[1:]):
                if hand_strength <= threshold:
                    strength_category = strength_labels[i]
                    break
            
            if strength_category:
                strength_action_counts[strength_category][action] += 1
        
        # Prepare plotting data
        categories = strength_labels
        actions = ['FOLD', 'CALL', 'RAISE', 'CHECK', 'ALL_IN']
        
        # Create stacked bar chart data
        bottom = np.zeros(len(categories))
        
        for action in actions:
            counts = [strength_action_counts[cat][action] for cat in categories]
            if sum(counts) > 0:  # Only plot actions with data
                ax.bar(categories, counts, bottom=bottom, 
                      label=action, color=action_colors.get(action, '#gray'),
                      alpha=0.8)
                bottom += counts
        
        # Set subplot title and labels
        ax.set_title(f'{display_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hand Strength', fontsize=12)
        ax.set_ylabel('Action Count', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add total decisions annotation - moved to top right
        total_decisions = len(data)
        ax.text(0.98, 0.98, f'Total: {total_decisions}', 
               transform=ax.transAxes, fontsize=11, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide extra subplots
    for idx in range(n_players, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Add legend - moved to avoid overlap
    if n_players > 0:
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                      ncol=len(handles), fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Leave more space for legend
    
    # Save figure
    filename = f"{experiment_name}_hand_strength_action_distribution.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
    plt.close()
    
    print(f"Hand strength-action distribution figure saved: {filename}")

def list_experiments():
    """List all available experiments"""
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        print("Experiments directory not found")
        return
    
    experiments = [d for d in os.listdir(experiments_dir) 
                  if os.path.isdir(os.path.join(experiments_dir, d))]
    
    if not experiments:
        print("No experiments found")
        return
    
    print("Available experiments:")
    for exp in sorted(experiments):
        exp_path = os.path.join(experiments_dir, exp)
        results_file = os.path.join(exp_path, 'results.json')
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    info = data.get('experiment_info', {})
                    total_battles = info.get('total_battles', 'Unknown')
                    completed_battles = info.get('completed_battles', 'Unknown')
                    status = info.get('status', 'Unknown')
                    print(f"  {exp} ({completed_battles}/{total_battles} battles, {status})")
            except:
                print(f"  {exp} (unable to read info)")
        else:
            print(f"  {exp} (no results file)")

def main():
    parser = argparse.ArgumentParser(description='Poker Game Evaluation Tool')
    parser.add_argument('--list', action='store_true', help='List all available experiments')
    parser.add_argument('--experiment', type=str, help='Name of experiment to analyze')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    if not args.experiment:
        print("Please specify experiment name to analyze, or use --list to see available experiments")
        return
    
    experiment_path = os.path.join("experiments", args.experiment)
    
    if not os.path.exists(experiment_path):
        print(f"Experiment does not exist: {args.experiment}")
        return
    
    try:
        # Load and analyze data
        print(f"Analyzing experiment: {args.experiment}")
        data = load_experiment_data(experiment_path)
        
        # Calculate metrics
        results = calculate_metrics(data)
        behavior_stats = calculate_behavior_stats(data)
        hand_strength_data = collect_hand_strength_actions(data)
        
        # Print tables
        print_results_table(results)
        print_behavior_stats_table(behavior_stats)
        
        # Generate figures
        generate_chips_over_rounds_figure(data, args.experiment)
        generate_hand_strength_action_figure(hand_strength_data, args.experiment)
        
        print(f"\nAnalysis completed!")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 