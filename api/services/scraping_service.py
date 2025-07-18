"""
Background service for periodic electricity price scraping.

This service runs the scraper at specified intervals to keep the database
up to date with the latest prices from the Elexys website.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks
from ..utils.scraping.elexys_scraper import ElexysElectricityScraper


class ScrapingService:
    """
    Background service for periodic electricity price scraping.

    Features:
    - Runs scraper at configurable intervals
    - Handles errors gracefully without stopping the service
    - Provides startup scraping for immediate data availability
    - Logs all activities for monitoring
    """

    def __init__(self, interval_hours: int = 1):
        """
        Initialize the scraping service.

        Args:
            interval_hours: Hours between scraping runs (default: 1 hour)
        """
        self.interval_hours = interval_hours
        self.logger = logging.getLogger(__name__)
        self.scraper = ElexysElectricityScraper()
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

    async def run_scraper_once(self) -> dict:
        """
        Run the scraper once and return statistics.

        Returns:
            Dictionary with scraping statistics
        """
        try:
            self.logger.info("🔄 Running electricity price scraper...")

            # Run scraper in executor to avoid blocking
            def sync_scrape():
                return self.scraper.scrape_and_update()

            stats = await asyncio.get_event_loop().run_in_executor(None, sync_scrape)

            if 'error' in stats:
                self.logger.error(f"❌ Scraper error: {stats['error']}")
            else:
                self.logger.info(
                    f"✅ Scraper completed: {stats.get('inserted', 0)} new, "
                    f"{stats.get('updated', 0)} updated, "
                    f"{stats.get('skipped_existing', 0)} skipped"
                )

            return stats

        except Exception as e:
            self.logger.error(f"❌ Scraper exception: {e}")
            return {'error': str(e)}

    async def start_periodic_scraping(self):
        """Start the periodic scraping background task."""
        if self.is_running:
            self.logger.warning("⚠️  Scraping service is already running")
            return

        self.is_running = True
        self.logger.info(
            f"🚀 Starting periodic scraping service (interval: {self.interval_hours}h)")

        # Run immediately on startup
        await self.run_scraper_once()

        # Start periodic task
        self._task = asyncio.create_task(self._periodic_scraping_loop())

    async def stop_periodic_scraping(self):
        """Stop the periodic scraping background task."""
        if not self.is_running:
            return

        self.is_running = False
        self.logger.info("🛑 Stopping periodic scraping service")

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _periodic_scraping_loop(self):
        """Internal method for the periodic scraping loop."""
        while self.is_running:
            try:
                # Wait for the specified interval
                # Convert hours to seconds
                await asyncio.sleep(self.interval_hours * 3600)

                if self.is_running:
                    await self.run_scraper_once()

            except asyncio.CancelledError:
                self.logger.info("📴 Periodic scraping loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in periodic scraping loop: {e}")
                # Continue running even if one iteration fails
                await asyncio.sleep(60)  # Wait 1 minute before retry


# Global scraping service instance
scraping_service = ScrapingService(interval_hours=6)


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan context manager for background tasks."""
    # Startup
    await scraping_service.start_periodic_scraping()

    try:
        yield
    finally:
        # Shutdown
        await scraping_service.stop_periodic_scraping()


def add_scraping_endpoint(app):
    """Add manual scraping endpoint to the FastAPI app."""

    @app.post("/api/scrape", tags=["System Information"])
    async def manual_scrape():
        """
        Manually trigger the electricity price scraper.

        This endpoint allows you to run the scraper on-demand without waiting
        for the scheduled interval.

        Returns:
            dict: Scraping statistics including new records, updates, and errors
        """
        stats = await scraping_service.run_scraper_once()
        return {
            "timestamp": datetime.now().isoformat(),
            "scraping_stats": stats,
            "message": "Manual scraping completed"
        }

    @app.get("/api/scraper/status", tags=["System Information"])
    async def scraper_status():
        """
        Get the current status of the periodic scraping service.

        Returns:
            dict: Service status information
        """
        return {
            "is_running": scraping_service.is_running,
            "interval_hours": scraping_service.interval_hours,
            "timestamp": datetime.now().isoformat()
        }
