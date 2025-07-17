"""
Command-line utility for running the Elexys electricity price scraper.

This script provides a simple interface to run the scraper with various options.
"""

import argparse
import sys
import os
from datetime import datetime

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elexys_scraper import ElexysElectricityScraper


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Scrape electricity prices from Elexys website and update the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scrape_prices.py                    # Run basic scraping
  python scrape_prices.py --verbose          # Run with detailed logging
  python scrape_prices.py --db-path custom.db  # Use custom database file

The scraper will:
1. Fetch the latest price data from Elexys
2. Check for existing records to avoid duplicates
3. Add only new records to the database
4. Calculate consumer prices with markup
5. Show a summary of the results
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

    args = parser.parse_args()

    # Create scraper instance
    try:
        scraper = ElexysElectricityScraper(db_path=args.db_path)

        # Configure logging level
        if args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        elif args.quiet:
            import logging
            logging.getLogger().setLevel(logging.ERROR)

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

        # Show current database status
        if not args.quiet:
            latest_timestamp = scraper.get_latest_db_timestamp()
            if latest_timestamp:
                print(f"📅 Latest data in database: {latest_timestamp}")
            else:
                print("📅 No existing data in database")

        # Run scraping
        if not args.quiet:
            print("\n🚀 Starting electricity price scraping...")

        stats = scraper.scrape_and_update()

        if not args.quiet:
            scraper.print_stats(stats)
        else:
            # Just print errors if any
            if 'error' in stats:
                print(f"❌ Error: {stats['error']}")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Scraping cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
