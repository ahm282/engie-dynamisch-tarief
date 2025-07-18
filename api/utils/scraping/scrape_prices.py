"""
Command-line utility for running the Elexys electricity price scraper.

This script provides a simple interface to run the scraper with various options.
It can run once or continuously in the background with scheduled intervals.
"""

from .elexys_scraper import ElexysElectricityScraper
import argparse
import sys
import os
import time
import signal
from datetime import datetime
import logging

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


def run_single_scrape(scraper, quiet=False):
    """Run a single scrape operation."""
    try:
        if not quiet:
            print("\n🚀 Starting electricity price scraping...")

        stats = scraper.scrape_and_update()

        if not quiet:
            scraper.print_stats(stats)
        else:
            # Just print errors if any
            if 'error' in stats:
                print(f"❌ Error: {stats['error']}")
                return False
        return True
    except Exception as e:
        print(f"❌ Scraping error: {e}")
        return False


def run_background_scheduler(scraper, interval_hours=6, quiet=False):
    """Run scraper continuously in background with scheduled intervals."""
    global shutdown_requested

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    interval_seconds = interval_hours * 3600  # Convert to seconds

    if not quiet:
        print(
            f"🕒 Starting background scheduler (every {interval_hours} hours)")
        print("Press Ctrl+C to stop gracefully")

    # Run initial scrape immediately
    if not quiet:
        print("\n🎯 Running initial scrape...")

    success = run_single_scrape(scraper, quiet)
    if not success and not quiet:
        print("⚠️  Initial scrape failed, but continuing with schedule...")

    last_run = datetime.now()
    next_run = datetime.now().timestamp() + interval_seconds

    if not quiet:
        next_run_time = datetime.fromtimestamp(next_run)
        print(f"⏰ Next scrape scheduled for: {next_run_time}")

    # Main scheduling loop
    while not shutdown_requested:
        try:
            current_time = datetime.now().timestamp()

            # Check if it's time for next scrape
            if current_time >= next_run:
                if not quiet:
                    print(f"\n🔄 Running scheduled scrape at {datetime.now()}")

                success = run_single_scrape(scraper, quiet)
                last_run = datetime.now()
                next_run = current_time + interval_seconds

                if not quiet:
                    next_run_time = datetime.fromtimestamp(next_run)
                    status = "✅ Success" if success else "❌ Failed"
                    print(f"{status} - Next scrape: {next_run_time}")

            # Sleep for 60 seconds before checking again
            time.sleep(60)

        except Exception as e:
            if not quiet:
                print(f"❌ Scheduler error: {e}")
            time.sleep(60)  # Wait before retrying

    if not quiet:
        print("\n✅ Background scheduler stopped gracefully")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Scrape electricity prices from Elexys website and update the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m utils.scraping.scrape_prices                    # Run basic scraping (once)
  python -m utils.scraping.scrape_prices --verbose          # Run with detailed logging
  python -m utils.scraping.scrape_prices --background       # Run continuously every 6 hours
  python -m utils.scraping.scrape_prices --background --interval 4  # Every 4 hours
  python -m utils.scraping.scrape_prices --db-path custom.db  # Use custom database file

The scraper will:
1. Fetch the latest price data from Elexys
2. Check for existing records to avoid duplicates
3. Add only new records to the database
4. Calculate consumer prices with markup
5. Show a summary of the results

Background mode:
- Runs continuously with scheduled intervals
- Handles shutdown signals gracefully (SIGINT, SIGTERM)
- Perfect for Docker containers or long-running processes
        """
    )

    parser.add_argument(
        '--db-path',
        type=str,
        help='Path to the SQLite database file (optional)',
        default=None
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )

    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check the latest data timestamp without scraping'
    )

    parser.add_argument(
        '--background', '-b',
        action='store_true',
        help='Run continuously in background with scheduled intervals'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=6,
        help='Interval in hours for background scraping (default: 6)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.background and args.check_only:
        print("❌ Error: --background and --check-only cannot be used together")
        sys.exit(1)

    if args.interval < 1:
        print("❌ Error: --interval must be at least 1 hour")
        sys.exit(1)

    # Create scraper instance
    try:
        scraper = ElexysElectricityScraper(db_path=args.db_path)

        # Configure logging level
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        elif args.quiet:
            logging.basicConfig(level=logging.ERROR)
        else:
            logging.basicConfig(level=logging.INFO)

        if args.check_only:
            # Just check latest timestamp
            latest_timestamp = scraper.get_latest_db_timestamp()
            if latest_timestamp:
                if not args.quiet:
                    print(f"📅 Latest data in database: {latest_timestamp}")
                    # Calculate how recent the data is
                    now = datetime.now()
                    age = now - latest_timestamp
                    hours_ago = age.total_seconds() / 3600
                    print(f"⏰ Data age: {hours_ago:.1f} hours ago")

                    if hours_ago < 2:
                        print("✅ Data is very recent")
                    elif hours_ago < 24:
                        print("🔶 Data is from today")
                    else:
                        print("🔴 Data is older than 24 hours")
            else:
                print("❌ No data found in database")
            return

        elif args.background:
            # Run in background mode with scheduling
            if not args.quiet:
                print(
                    f"🔄 Starting background scraper (every {args.interval} hours)")
                latest_timestamp = scraper.get_latest_db_timestamp()
                if latest_timestamp:
                    print(f"📅 Latest data in database: {latest_timestamp}")
                else:
                    print("📅 No existing data in database")

            run_background_scheduler(scraper, args.interval, args.quiet)

        else:
            # Single run mode
            # Show current database status
            if not args.quiet:
                latest_timestamp = scraper.get_latest_db_timestamp()
                if latest_timestamp:
                    print(f"📅 Latest data in database: {latest_timestamp}")
                else:
                    print("📅 No existing data in database")

            # Run single scraping
            success = run_single_scrape(scraper, args.quiet)
            if not success:
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Scraping cancelled by user")
        sys.exit(0)  # Exit gracefully for Ctrl+C
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
