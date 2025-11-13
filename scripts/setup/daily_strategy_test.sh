#!/bin/bash
# Daily Crypto Strategy Testing Script
#
# This script runs daily backtests on all crypto strategies
# and saves results with timestamps.
#
# Usage:
#   ./daily_strategy_test.sh
#
# To schedule daily execution with cron:
#   crontab -e
#   Add: 0 9 * * * /path/to/daily_strategy_test.sh
#   (Runs every day at 9 AM)

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname $(dirname "$SCRIPT_DIR"))"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create logs directory
mkdir -p "$PROJECT_ROOT/backtest/logs"

# Run the strategy tester
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/backtest/logs/strategy_test_$TIMESTAMP.log"

echo "================================================================"
echo "Daily Crypto Strategy Test"
echo "Started: $(date)"
echo "================================================================"

python "$PROJECT_ROOT/src/strategies/comprehensive_strategy_tester.py" --asset-class multi 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================"
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Log saved to: $LOG_FILE"
echo "================================================================"

exit $EXIT_CODE
