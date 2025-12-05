# =============================================================================
# GAME CONFIGURATION CONSTANTS
# =============================================================================

# Game settings
NUM_BATTLES = 10                    # Total number of battles to run
ROUNDS_PER_BATTLE = 50             # Number of rounds per battle
INITIAL_STACK = 1000              # Initial stack for each player
SMALL_BLIND_AMOUNT = 5             # Small blind amount

# System settings
VERBOSE_LEVEL = 1                  # Game verbosity level (0=silent, 1=verbose)
EXPERIMENTS_DIR = "experiments"    # Directory for experiment results
LOG_LEVEL = "INFO"                 # Logging level

# Player aliases will be retrieved from player classes dynamically

# =============================================================================

from pypokerengine.api.game import setup_config, start_poker
from players.gpt_player import GPTPlayer
from players.llama_player import LlamaPlayer
from players.monte_player import MontePlayer
from players.deepSeek_player import DeepSeekPlayer
from players.claude_player import ClaudePlayer
from players.qwen_player import QwenPlayer
from players.gemini_player import GeminiPlayer
import json
import os
import logging
import sys
from datetime import datetime, timedelta
import uuid
import shutil
import io
import time
from contextlib import redirect_stdout

def get_player_aliases():
    """Get player aliases from their respective classes"""
    return {
        "gpt": GPTPlayer.PLAYER_ALIAS,
        "llama": LlamaPlayer.PLAYER_ALIAS, 
        "claude": ClaudePlayer.PLAYER_ALIAS,
        "monte": MontePlayer.PLAYER_ALIAS,
        "deepseek": DeepSeekPlayer.PLAYER_ALIAS,
        "qwen": QwenPlayer.PLAYER_ALIAS,
        "gemini": GeminiPlayer.PLAYER_ALIAS
    }

class TeeOutput:
    """Custom output class that writes to both console and file"""
    def __init__(self, console, file_handle):
        self.console = console
        self.file = file_handle
    
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate writing
    
    def flush(self):
        self.console.flush()
        self.file.flush()

class ExperimentManager:
    """Manages poker game experiments with logging and incremental saving"""
    
    def __init__(self, experiment_name=None):
        # Generate experiment ID and setup directories
        self.experiment_id = experiment_name or f"poker_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        self.exp_dir = os.path.join(EXPERIMENTS_DIR, self.experiment_id)
        self.results_file = os.path.join(self.exp_dir, "results.json")
        self.config_file = os.path.join(self.exp_dir, "config.json")
        self.log_file = os.path.join(self.exp_dir, "experiment.log")
        self.resume_file = os.path.join(self.exp_dir, "resume_state.json")
        
        # Create experiment directory
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize experiment state
        self.results = {"battles": [], "experiment_info": {}}
        self.resume_state = {"completed_battles": 0, "status": "running"}
        
        # Time tracking for progress estimation
        self.experiment_start_time = time.time()
        self.battle_times = []  # Store duration of each completed battle
        self.current_battle_start_time = None
        
        # Save experiment configuration
        self.save_config()
        
        # Setup output redirection for capturing all game output
        self.game_log_file = os.path.join(self.exp_dir, "game_output.log")
        self.original_stdout = sys.stdout
        
        self.logger.info(f"Experiment initialized: {self.experiment_id}")
    
    def setup_logging(self):
        """Setup logging to both file and console"""
        # Create logger
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(getattr(logging, LOG_LEVEL))
        
        # Remove existing handlers to avoid duplication
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def save_config(self):
        """Save experiment configuration"""
        config = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "settings": {
                "num_battles": NUM_BATTLES,
                "rounds_per_battle": ROUNDS_PER_BATTLE,
                "initial_stack": INITIAL_STACK,
                "small_blind_amount": SMALL_BLIND_AMOUNT,
                "verbose_level": VERBOSE_LEVEL
            },
            "players": get_player_aliases()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_resume_state(self):
        """Load resume state if exists"""
        if os.path.exists(self.resume_file):
            with open(self.resume_file, 'r') as f:
                self.resume_state = json.load(f)
            self.logger.info(f"Resuming from battle {self.resume_state['completed_battles'] + 1}")
            return True
        return False
    
    def save_resume_state(self):
        """Save current resume state"""
        with open(self.resume_file, 'w') as f:
            json.dump(self.resume_state, f, indent=2)
    
    def load_existing_results(self):
        """Load existing results if resuming"""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            
            # Reset experiment start time for resumed experiments
            # This gives better time estimates for remaining work
            self.experiment_start_time = time.time()
    
    def save_battle_result(self, battle_num, battle_result):
        """Save individual battle result incrementally"""
        # Load existing results if file exists
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
        
        # Ensure battles list exists and has correct length
        while len(self.results["battles"]) <= battle_num:
            self.results["battles"].append(None)
        
        # Add battle result
        self.results["battles"][battle_num] = battle_result
        
        # Update experiment info
        self.results["experiment_info"] = {
            "experiment_id": self.experiment_id,
            "total_battles": NUM_BATTLES,
            "completed_battles": battle_num + 1,
            "last_updated": datetime.now().isoformat(),
            "status": "completed" if battle_num + 1 >= NUM_BATTLES else "running"
        }
        
        # Save to file
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Update resume state
        self.resume_state["completed_battles"] = battle_num + 1
        self.resume_state["status"] = self.results["experiment_info"]["status"]
        self.save_resume_state()
        
        self.logger.info(f"Battle {battle_num + 1} results saved")
    
    def start_game_logging(self):
        """Start redirecting stdout to both console and game log file"""
        self.game_log_handle = open(self.game_log_file, 'a', encoding='utf-8')
        self.tee_output = TeeOutput(self.original_stdout, self.game_log_handle)
        sys.stdout = self.tee_output
    
    def stop_game_logging(self):
        """Stop redirecting stdout and close game log file"""
        try:
            if hasattr(self, 'tee_output'):
                sys.stdout = self.original_stdout
            if hasattr(self, 'game_log_handle') and self.game_log_handle:
                self.game_log_handle.close()
                self.game_log_handle = None
        except Exception as e:
            # Ensure we can still restore stdout even if file operations fail
            sys.stdout = self.original_stdout
            self.logger.warning(f"Error stopping game logging: {e}")
    
    def start_battle_timer(self):
        """Start timing for current battle"""
        self.current_battle_start_time = time.time()
    
    def end_battle_timer(self):
        """End timing for current battle and calculate statistics"""
        if self.current_battle_start_time is not None:
            battle_duration = time.time() - self.current_battle_start_time
            self.battle_times.append(battle_duration)
            self.current_battle_start_time = None
            return battle_duration
        return 0
    
    def format_time(self, seconds):
        """Format seconds into human-readable time string"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def get_time_estimates(self, current_battle_num):
        """Calculate time estimates for remaining battles"""
        if not self.battle_times:
            return None, None, None
        
        # Calculate average battle time
        avg_battle_time = sum(self.battle_times) / len(self.battle_times)
        
        # Calculate remaining battles
        remaining_battles = NUM_BATTLES - (current_battle_num + 1)
        
        # Estimate remaining time
        estimated_remaining_time = remaining_battles * avg_battle_time
        
        # Calculate total elapsed time
        total_elapsed = time.time() - self.experiment_start_time
        
        # Estimate total experiment time
        estimated_total_time = total_elapsed + estimated_remaining_time
        
        return avg_battle_time, estimated_remaining_time, estimated_total_time
    
    def log_progress_stats(self, battle_num, battle_duration):
        """Log detailed progress statistics"""
        completed_battles = battle_num + 1
        progress_percent = (completed_battles / NUM_BATTLES) * 100
        
        self.logger.info(f"📊 Progress: {completed_battles}/{NUM_BATTLES} battles ({progress_percent:.1f}%)")
        self.logger.info(f"⏱️  Battle {completed_battles} duration: {self.format_time(battle_duration)}")
        
        if len(self.battle_times) >= 2:  # Need at least 2 battles for good estimate
            avg_time, remaining_time, total_time = self.get_time_estimates(battle_num)
            
            self.logger.info(f"📈 Average battle time: {self.format_time(avg_time)}")
            self.logger.info(f"⏳ Estimated remaining time: {self.format_time(remaining_time)}")
            
            # Calculate ETA
            if remaining_time is not None:
                eta = datetime.now() + timedelta(seconds=remaining_time)
                self.logger.info(f"🎯 Estimated completion: {eta.strftime('%H:%M:%S')}")
        
        # Log total elapsed time
        total_elapsed = time.time() - self.experiment_start_time
        self.logger.info(f"🕐 Total elapsed time: {self.format_time(total_elapsed)}")
    
    def get_starting_battle(self):
        """Get the battle number to start from (for resume functionality)"""
        if self.load_resume_state():
            self.load_existing_results()
            return self.resume_state["completed_battles"]
        return 0

def run_poker_game(rounds=ROUNDS_PER_BATTLE, initial_stack=INITIAL_STACK, small_blind=SMALL_BLIND_AMOUNT):
    """Run a single poker game and return detailed results"""
    # Initialize players
    monte = MontePlayer()
    gpt = GPTPlayer()
    llama = LlamaPlayer()
    claude = ClaudePlayer()
    deepseek = DeepSeekPlayer()
    qwen = QwenPlayer()
    gemini = GeminiPlayer()
    
    # Get player aliases
    player_names = get_player_aliases()
    
    # Setup game configuration
    config = setup_config(max_round=rounds, initial_stack=initial_stack, small_blind_amount=small_blind)
    config.register_player(name=player_names["gpt"], algorithm=gpt)
    config.register_player(name=player_names["llama"], algorithm=llama)
    config.register_player(name=player_names["claude"], algorithm=claude)
    config.register_player(name=player_names["monte"], algorithm=monte)
    config.register_player(name=player_names["deepseek"], algorithm=deepseek)
    config.register_player(name=player_names["qwen"], algorithm=qwen)
    config.register_player(name=player_names["gemini"], algorithm=gemini)
    
    # Start game
    game_result = start_poker(config, verbose=VERBOSE_LEVEL)
    
    # Collect basic results
    results = {
        "game_info": {
            "timestamp": datetime.now().isoformat(),
            "rounds": rounds,
            "initial_stack": initial_stack,
            "small_blind": small_blind
        },
        "players": {},
        "reasoning_history": {}  # New: collect reasoning data
    }
    
    # Collect player statistics and reasoning history
    player_algorithms = [
        (player_names["llama"], llama), 
        (player_names["monte"], monte),  # Now uses consistent naming
        (player_names["claude"], claude), 
        (player_names["gpt"], gpt), 
        (player_names["deepseek"], deepseek), 
        (player_names["qwen"], qwen),
        (player_names["gemini"], gemini)
    ]
    
    for player_name, algorithm in player_algorithms:
        # Get performance stats
        stats = algorithm.get_performance_stats()
        results["players"][player_name] = stats
        
        # Get reasoning history if available (LLM players only)
        if hasattr(algorithm, 'reasoning_history'):
            results["reasoning_history"][player_name] = algorithm.reasoning_history
    
    return results

def main():
    """Main experiment execution function"""
    # Initialize experiment manager
    exp_manager = ExperimentManager()
    
    exp_manager.logger.info("="*60)
    exp_manager.logger.info(f"Starting poker experiment: {exp_manager.experiment_id}")
    exp_manager.logger.info(f"Configuration: {NUM_BATTLES} battles, {ROUNDS_PER_BATTLE} rounds each")
    exp_manager.logger.info(f"Initial stack: {INITIAL_STACK}, Small blind: {SMALL_BLIND_AMOUNT}")
    exp_manager.logger.info(f"Results will be saved to: {exp_manager.exp_dir}")
    exp_manager.logger.info("="*60)
    
    # Check for resume
    start_battle = exp_manager.get_starting_battle()
    if start_battle > 0:
        exp_manager.logger.info(f"Resuming experiment from battle {start_battle + 1}")
    
    try:
        # Run battles
        for i in range(start_battle, NUM_BATTLES):
            exp_manager.logger.info(f"Starting BATTLE {i+1}/{NUM_BATTLES}")
            exp_manager.logger.info("="*50)
            
            # Start battle timing
            exp_manager.start_battle_timer()
            
            # Start game output logging
            exp_manager.start_game_logging()
            
            try:
                # Run single battle
                battle_result = run_poker_game(
                    rounds=ROUNDS_PER_BATTLE, 
                    initial_stack=INITIAL_STACK, 
                    small_blind=SMALL_BLIND_AMOUNT
                )
            finally:
                # Ensure we stop logging even if game fails
                exp_manager.stop_game_logging()
            
            # End battle timing and get duration
            battle_duration = exp_manager.end_battle_timer()
            
            # Save battle result immediately
            exp_manager.save_battle_result(i, battle_result)
            
            # Log progress statistics with time estimates
            exp_manager.log_progress_stats(i, battle_duration)
            exp_manager.logger.info("-"*50)
        
        exp_manager.logger.info("="*60)
        exp_manager.logger.info("All battles completed successfully!")
        exp_manager.logger.info(f"Results saved in: {exp_manager.exp_dir}")
        exp_manager.logger.info("="*60)
        
        # Clean up resume file on successful completion
        if os.path.exists(exp_manager.resume_file):
            os.remove(exp_manager.resume_file)
    
    except KeyboardInterrupt:
        exp_manager.logger.warning("Experiment interrupted by user")
        exp_manager.logger.info(f"Progress saved. Resume with: python run_game.py --resume {exp_manager.experiment_id}")
    except Exception as e:
        exp_manager.logger.error(f"Experiment failed with error: {str(e)}")
        exp_manager.logger.info(f"Progress saved. Resume with: python run_game.py --resume {exp_manager.experiment_id}")
        raise

if __name__ == "__main__":
    import argparse
    
    # Add command line argument parsing for resume functionality
    parser = argparse.ArgumentParser(description="Run poker game experiments")
    parser.add_argument("--resume", type=str, help="Resume experiment with given ID")
    args = parser.parse_args()
    
    if args.resume:
        # Resume existing experiment
        if not os.path.exists(os.path.join(EXPERIMENTS_DIR, args.resume)):
            print(f"Error: Experiment directory {args.resume} not found")
            sys.exit(1)
        
        # Initialize with existing experiment ID
        exp_manager = ExperimentManager(args.resume)
        
        # Run main experiment logic
        exp_manager.logger.info("="*60)
        exp_manager.logger.info(f"Resuming poker experiment: {exp_manager.experiment_id}")
        exp_manager.logger.info("="*60)
        
        # Check for resume
        start_battle = exp_manager.get_starting_battle()
        exp_manager.logger.info(f"Resuming from battle {start_battle + 1}")
        
        try:
            # Run remaining battles
            for i in range(start_battle, NUM_BATTLES):
                exp_manager.logger.info(f"Starting BATTLE {i+1}/{NUM_BATTLES}")
                exp_manager.logger.info("="*50)
                
                # Start battle timing
                exp_manager.start_battle_timer()
                
                # Start game output logging
                exp_manager.start_game_logging()
                
                try:
                    # Run single battle
                    battle_result = run_poker_game(
                        rounds=ROUNDS_PER_BATTLE, 
                        initial_stack=INITIAL_STACK, 
                        small_blind=SMALL_BLIND_AMOUNT
                    )
                finally:
                    # Ensure we stop logging even if game fails
                    exp_manager.stop_game_logging()
                
                # End battle timing and get duration
                battle_duration = exp_manager.end_battle_timer()
                
                # Save battle result immediately
                exp_manager.save_battle_result(i, battle_result)
                
                # Log progress statistics with time estimates
                exp_manager.log_progress_stats(i, battle_duration)
                exp_manager.logger.info("-"*50)
            
            exp_manager.logger.info("="*60)
            exp_manager.logger.info("All battles completed successfully!")
            exp_manager.logger.info(f"Results saved in: {exp_manager.exp_dir}")
            exp_manager.logger.info("="*60)
            
            # Clean up resume file on successful completion
            if os.path.exists(exp_manager.resume_file):
                os.remove(exp_manager.resume_file)
        
        except KeyboardInterrupt:
            exp_manager.logger.warning("Experiment interrupted by user")
            exp_manager.logger.info(f"Progress saved. Resume with: python run_game.py --resume {exp_manager.experiment_id}")
        except Exception as e:
            exp_manager.logger.error(f"Experiment failed with error: {str(e)}")
            exp_manager.logger.info(f"Progress saved. Resume with: python run_game.py --resume {exp_manager.experiment_id}")
            raise
    else:
        # Start new experiment
        main()
    