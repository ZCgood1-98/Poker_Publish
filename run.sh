#!/bin/bash

# =============================================================================
# Poker Experiment Watchdog Script
# =============================================================================
# This script provides automatic monitoring and recovery for poker experiments.
# It will automatically resume experiments if they fail due to errors.
# Usage: bash run.sh [--test]
# --test: Use test_watchdog.py instead of run_game.py for testing
# =============================================================================

set -e  # Exit on any error (will be handled by our watchdog)

# Configuration
PYTHON_SCRIPT="run_game.py"
TEST_SCRIPT="test_watchdog.py"
MAX_RETRY_ATTEMPTS=5
RETRY_DELAY=10  # seconds
EXPERIMENTS_DIR="experiments"
LOG_FILE="watchdog.log"

# Check for test mode
if [ "$1" = "--test" ]; then
    PYTHON_SCRIPT="${TEST_SCRIPT}"
    MAX_RETRY_ATTEMPTS=3  # Fewer attempts for testing
    RETRY_DELAY=2  # Shorter delay for testing
    echo "Running in TEST MODE"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[${timestamp}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

# Print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
    log_message "INFO" "${message}"
}

# Get the most recent incomplete experiment ID
get_incomplete_experiment() {
    if [ ! -d "${EXPERIMENTS_DIR}" ]; then
        return 1
    fi
    
    # Find experiments with resume_state.json (incomplete experiments)
    local resume_files=($(find "${EXPERIMENTS_DIR}" -name "resume_state.json" -type f 2>/dev/null))
    
    if [ ${#resume_files[@]} -eq 0 ]; then
        return 1
    fi
    
    # Get the most recent one
    local latest_file=""
    local latest_time=0
    
    for file in "${resume_files[@]}"; do
        local file_time=$(stat -c %Y "${file}" 2>/dev/null || echo 0)
        if [ "${file_time}" -gt "${latest_time}" ]; then
            latest_time="${file_time}"
            latest_file="${file}"
        fi
    done
    
    if [ -n "${latest_file}" ]; then
        # Extract experiment ID from path
        local exp_id=$(dirname "${latest_file}")
        exp_id=$(basename "${exp_id}")
        echo "${exp_id}"
        return 0
    fi
    
    return 1
}

# Check if experiment is complete
is_experiment_complete() {
    local exp_id=$1
    local resume_file="${EXPERIMENTS_DIR}/${exp_id}/resume_state.json"
    
    if [ ! -f "${resume_file}" ]; then
        return 0  # No resume file means either complete or never started
    fi
    
    # Check status in resume file
    local status=$(python3 -c "
import json
import sys
try:
    with open('${resume_file}', 'r') as f:
        data = json.load(f)
    print(data.get('status', 'unknown'))
except:
    print('unknown')
" 2>/dev/null)
    
    if [ "${status}" = "completed" ]; then
        return 0  # Complete
    else
        return 1  # Incomplete
    fi
}

# Run experiment with automatic retry
run_experiment_with_retry() {
    local attempt=1
    local experiment_id=""
    
    while [ ${attempt} -le ${MAX_RETRY_ATTEMPTS} ]; do
        print_status "${BLUE}" "=========================================="
        print_status "${BLUE}" "Attempt ${attempt}/${MAX_RETRY_ATTEMPTS}"
        print_status "${BLUE}" "=========================================="
        
        # Check for incomplete experiments first
        if [ ${attempt} -gt 1 ] || [ -n "$(get_incomplete_experiment 2>/dev/null)" ]; then
            experiment_id=$(get_incomplete_experiment 2>/dev/null || echo "")
            if [ -n "${experiment_id}" ]; then
                if ! is_experiment_complete "${experiment_id}"; then
                    print_status "${YELLOW}" "Found incomplete experiment: ${experiment_id}"
                    print_status "${YELLOW}" "Attempting to resume..."
                    
                    # Try to resume
                    if python3 "${PYTHON_SCRIPT}" --resume "${experiment_id}"; then
                        print_status "${GREEN}" "Experiment resumed and completed successfully!"
                        log_message "SUCCESS" "Experiment ${experiment_id} completed"
                        return 0
                    else
                        local exit_code=$?
                        print_status "${RED}" "Resume attempt failed with exit code: ${exit_code}"
                        log_message "ERROR" "Resume failed for experiment ${experiment_id} with exit code ${exit_code}"
                    fi
                else
                    print_status "${GREEN}" "Experiment ${experiment_id} is already complete!"
                    return 0
                fi
            fi
        fi
        
        # If no incomplete experiment or resume failed, start new experiment
        if [ ${attempt} -eq 1 ]; then
            print_status "${BLUE}" "Starting new poker experiment..."
        else
            print_status "${YELLOW}" "Starting new experiment (previous attempts failed)..."
        fi
        
        # Run the experiment
        if python3 "${PYTHON_SCRIPT}"; then
            print_status "${GREEN}" "Experiment completed successfully!"
            log_message "SUCCESS" "New experiment completed successfully"
            return 0
        else
            local exit_code=$?
            print_status "${RED}" "Experiment failed with exit code: ${exit_code}"
            log_message "ERROR" "Experiment attempt ${attempt} failed with exit code ${exit_code}"
            
            # Check if we should retry
            if [ ${attempt} -lt ${MAX_RETRY_ATTEMPTS} ]; then
                print_status "${YELLOW}" "Waiting ${RETRY_DELAY} seconds before retry..."
                sleep ${RETRY_DELAY}
            fi
        fi
        
        attempt=$((attempt + 1))
    done
    
    print_status "${RED}" "All retry attempts exhausted. Experiment failed."
    log_message "FAILURE" "All ${MAX_RETRY_ATTEMPTS} attempts failed"
    return 1
}

# Cleanup function
cleanup() {
    log_message "INFO" "Watchdog script terminated"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main execution
main() {
    print_status "${GREEN}" "=========================================="
    print_status "${GREEN}" "Poker Experiment Watchdog Started"
    print_status "${GREEN}" "=========================================="
    
    log_message "START" "Watchdog script started"
    log_message "CONFIG" "Max retry attempts: ${MAX_RETRY_ATTEMPTS}"
    log_message "CONFIG" "Retry delay: ${RETRY_DELAY} seconds"
    log_message "CONFIG" "Python script: ${PYTHON_SCRIPT}"
    
    # Check if Python script exists
    if [ ! -f "${PYTHON_SCRIPT}" ]; then
        print_status "${RED}" "Error: ${PYTHON_SCRIPT} not found!"
        log_message "ERROR" "Python script ${PYTHON_SCRIPT} not found"
        exit 1
    fi
    
    # Check Python environment
    if ! python3 -c "import sys; print('Python', sys.version)" 2>/dev/null; then
        print_status "${RED}" "Error: Python3 not available!"
        log_message "ERROR" "Python3 not available"
        exit 1
    fi
    
    # Create experiments directory if it doesn't exist
    mkdir -p "${EXPERIMENTS_DIR}"
    
    # Run the experiment with retry logic
    if run_experiment_with_retry; then
        print_status "${GREEN}" "=========================================="
        print_status "${GREEN}" "All experiments completed successfully!"
        print_status "${GREEN}" "Check the experiments/ directory for results"
        print_status "${GREEN}" "=========================================="
        log_message "COMPLETE" "All experiments completed successfully"
        exit 0
    else
        print_status "${RED}" "=========================================="
        print_status "${RED}" "Experiments failed after all retry attempts"
        print_status "${RED}" "Check ${LOG_FILE} for detailed logs"
        print_status "${RED}" "=========================================="
        log_message "FAILED" "Experiments failed after all retry attempts"
        exit 1
    fi
}

# Check if script is being sourced or executed
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi 