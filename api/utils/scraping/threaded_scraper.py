"""
Threaded web scraper for Elexys electricity price data.

This module provides a simple threaded scraping solution that runs the 
ElexysElectricityScraper in a background thread with configurable intervals.
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import signal
import sys

from .elexys_scraper import ElexysElectricityScraper


class ThreadedElexysScrapingService:
    """
    Simple threaded service for continuous electricity price scraping.

    Features:
    - Runs scraping in background thread
    - Configurable scraping intervals
    - Graceful shutdown handling
    - Basic logging and error handling
    - Thread-safe operations
    """

    def __init__(
        self,
        scrape_interval_minutes: int = 60,
        db_path: str = None,
        max_retries: int = 3,
        retry_delay_seconds: int = 300,  # 5 minutes
        log_level: str = "INFO"
    ):
        """
        Initialize the threaded scraping service.

        Args:
            scrape_interval_minutes: Minutes between scraping attempts (default: 60)
            db_path: Optional path to the database file
            max_retries: Maximum retry attempts on failure (default: 3)
            retry_delay_seconds: Delay between retries in seconds (default: 300)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.scrape_interval = scrape_interval_minutes * 60  # Convert to seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay_seconds

        # Initialize the scraper
        self.scraper = ElexysElectricityScraper(db_path=db_path)

        # Threading components
        self.thread = None
        self.shutdown_event = threading.Event()

        # Status tracking
        self.is_running = False
        self.last_scrape_time = None
        self.last_scrape_status = None
        self.total_scrapes = 0
        self.successful_scrapes = 0
        self.failed_scrapes = 0

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def start(self) -> bool:
        """
        Start the threaded scraping service.

        Returns:
            bool: True if started successfully, False if already running
        """
        if self.is_running:
            self.logger.warning("Scraping service is already running")
            return False

        self.logger.info(
            f"Starting threaded scraping service (interval: {self.scrape_interval/60:.1f} minutes)")
        self.is_running = True
        self.shutdown_event.clear()

        # Create and start the background thread
        self.thread = threading.Thread(
            target=self._scraping_loop,
            name="ElexysScrapingThread",
            daemon=True
        )
        self.thread.start()

        self.logger.info("Scraping service started successfully")
        return True

    def stop(self, timeout: int = 30) -> bool:
        """
        Stop the threaded scraping service gracefully.

        Args:
            timeout: Maximum seconds to wait for shutdown (default: 30)

        Returns:
            bool: True if stopped successfully, False if timeout occurred
        """
        if not self.is_running:
            self.logger.warning("Scraping service is not running")
            return True

        self.logger.info("Stopping threaded scraping service...")
        self.shutdown_event.set()
        self.is_running = False

        # Wait for the thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                self.logger.error(
                    f"Thread did not stop within {timeout} seconds")
                return False

        self.logger.info("Scraping service stopped successfully")
        return True

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the scraping service.

        Returns:
            dict: Status information including statistics and timing
        """
        next_scrape_time = None
        if self.last_scrape_time and self.is_running:
            next_scrape_time = self.last_scrape_time + \
                timedelta(seconds=self.scrape_interval)

        return {
            "is_running": self.is_running,
            "scrape_interval_minutes": self.scrape_interval / 60,
            "last_scrape_time": self.last_scrape_time.isoformat() if self.last_scrape_time else None,
            "next_scrape_time": next_scrape_time.isoformat() if next_scrape_time else None,
            "last_scrape_status": self.last_scrape_status,
            "total_scrapes": self.total_scrapes,
            "successful_scrapes": self.successful_scrapes,
            "failed_scrapes": self.failed_scrapes,
            "success_rate": (self.successful_scrapes / self.total_scrapes * 100) if self.total_scrapes > 0 else 0
        }

    def force_scrape(self) -> Dict[str, Any]:
        """
        Force an immediate scrape operation.

        Returns:
            dict: Results of the scrape operation
        """
        self.logger.info("Forcing immediate scrape operation")
        return self._perform_scrape()

    def _scraping_loop(self):
        """
        Main scraping loop that runs in the background thread.
        """
        self.logger.info("Scraping loop started")

        # Perform initial scrape immediately
        if not self.shutdown_event.is_set():
            self._perform_scrape()

        while not self.shutdown_event.is_set():
            try:
                # Wait for the next scrape interval or shutdown signal
                if self.shutdown_event.wait(timeout=self.scrape_interval):
                    break  # Shutdown was requested

                # Perform scraping if not shutting down
                if not self.shutdown_event.is_set():
                    self._perform_scrape()

            except Exception as e:
                self.logger.error(f"Unexpected error in scraping loop: {e}")
                self.failed_scrapes += 1
                self.last_scrape_status = "error"

                # Wait before continuing
                if self.shutdown_event.wait(timeout=min(self.retry_delay, 60)):
                    break

        self.logger.info("Scraping loop ended")

    def _perform_scrape(self) -> Dict[str, Any]:
        """
        Perform a single scrape operation with retry logic.

        Returns:
            dict: Results of the scrape operation
        """
        self.last_scrape_time = datetime.now()
        self.total_scrapes += 1

        for attempt in range(self.max_retries):
            try:
                self.logger.info(
                    f"Starting scrape attempt {attempt + 1}/{self.max_retries}")

                # Perform the actual scraping
                stats = self.scraper.scrape_and_update()

                # Update success statistics
                self.successful_scrapes += 1
                self.last_scrape_status = "success"

                result = {
                    "status": "success",
                    "timestamp": self.last_scrape_time.isoformat(),
                    "attempt": attempt + 1,
                    "stats": stats,
                    "message": f"Scraping completed successfully. Added {stats.get('inserted', 0)} new records."
                }

                self.logger.info(result["message"])
                return result

            except Exception as e:
                self.logger.error(f"Scrape attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    # Wait before retrying (unless it's the last attempt)
                    retry_wait = min(self.retry_delay, 60 *
                                     (attempt + 1))  # Progressive delay
                    self.logger.info(
                        f"Waiting {retry_wait} seconds before retry...")

                    if self.shutdown_event.wait(timeout=retry_wait):
                        return {"status": "cancelled", "message": "Scraping cancelled during retry"}

        # All attempts failed
        self.failed_scrapes += 1
        self.last_scrape_status = "failed"

        error_result = {
            "status": "failed",
            "timestamp": self.last_scrape_time.isoformat(),
            "attempts": self.max_retries,
            "message": f"All {self.max_retries} scrape attempts failed"
        }

        self.logger.error(error_result["message"])
        return error_result

    def _signal_handler(self, signum, frame):
        """
        Handle system signals for graceful shutdown.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.logger.info(
            f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def start_background_scraping(
    interval_minutes: int = 60,
    db_path: str = None,
    log_level: str = "INFO"
) -> ThreadedElexysScrapingService:
    """
    Start background scraping with simple configuration.

    Args:
        interval_minutes: Minutes between scrapes (default: 60)
        db_path: Optional database path
        log_level: Logging level (default: INFO)

    Returns:
        ThreadedElexysScrapingService: The running service instance
    """
    service = ThreadedElexysScrapingService(
        scrape_interval_minutes=interval_minutes,
        db_path=db_path,
        log_level=log_level
    )

    service.start()
    return service


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Threaded Elexys electricity price scraper")
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Scraping interval in minutes (default: 60)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to database file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon (no user interaction)"
    )

    args = parser.parse_args()

    print(f"Starting threaded scraping service...")
    print(f"Interval: {args.interval} minutes")
    print(f"Database: {args.db_path or 'default'}")
    print(f"Log level: {args.log_level}")

    try:
        service = ThreadedElexysScrapingService(
            scrape_interval_minutes=args.interval,
            db_path=args.db_path,
            log_level=args.log_level
        )

        service.start()

        if args.daemon:
            # Run as daemon
            print("Running as daemon. Use Ctrl+C to stop.")
            while service.is_running:
                time.sleep(5)
        else:
            # Interactive mode
            print("Service running. Commands:")
            print("  status - Show service status")
            print("  scrape - Force immediate scrape")
            print("  stop   - Stop the service")
            print("  quit   - Stop service and exit")

            while service.is_running:
                try:
                    command = input("\n> ").strip().lower()

                    if command == "status":
                        status = service.get_status()
                        print(f"Running: {status['is_running']}")
                        print(
                            f"Interval: {status['scrape_interval_minutes']} minutes")
                        print(f"Last scrape: {status['last_scrape_time']}")
                        print(f"Next scrape: {status['next_scrape_time']}")
                        print(f"Total scrapes: {status['total_scrapes']}")
                        print(f"Success rate: {status['success_rate']:.1f}%")

                    elif command == "scrape":
                        print("Forcing immediate scrape...")
                        result = service.force_scrape()
                        print(
                            f"Result: {result['status']} - {result['message']}")

                    elif command in ["stop", "quit"]:
                        print("Stopping service...")
                        service.stop()
                        break

                    elif command == "help":
                        print("Commands: status, scrape, stop, quit, help")

                    else:
                        print("Unknown command. Type 'help' for available commands.")

                except (EOFError, KeyboardInterrupt):
                    print("\nShutdown requested...")
                    service.stop()
                    break

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")

    print("Service stopped.")
