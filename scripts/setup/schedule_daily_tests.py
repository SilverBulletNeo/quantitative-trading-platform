"""
Schedule Daily Strategy Tests

Sets up automated daily testing using Python's schedule library.
Runs in the background and executes strategy tests at specified time.

Usage:
    python schedule_daily_tests.py

Requirements:
    pip install schedule
"""

import schedule
import time
import subprocess
import os
from datetime import datetime


def run_strategy_tests():
    """Execute daily strategy tests"""
    print(f"\n{'='*80}")
    print(f"Running Daily Strategy Tests - {datetime.now()}")
    print(f"{'='*80}\n")

    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # Run the tester
    tester_path = os.path.join(project_root, 'src', 'strategies', 'crypto_strategy_tester.py')

    try:
        result = subprocess.run(
            ['python', tester_path],
            cwd=project_root,
            capture_output=True,
            text=True
        )

        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

        print(f"\nTest completed with exit code: {result.returncode}")

    except Exception as e:
        print(f"Error running tests: {e}")

    print(f"\n{'='*80}\n")


def main():
    """Set up and run the scheduler"""
    print("=" * 80)
    print("DAILY CRYPTO STRATEGY TEST SCHEDULER")
    print("=" * 80)
    print()
    print("Scheduling daily tests at 09:00 AM")
    print("Press Ctrl+C to stop the scheduler")
    print()

    # Schedule daily test at 9 AM
    schedule.every().day.at("09:00").do(run_strategy_tests)

    # Optional: Run test immediately on start
    print("Running initial test...")
    run_strategy_tests()

    # Keep the scheduler running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\nScheduler stopped by user")


if __name__ == "__main__":
    main()
