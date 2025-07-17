"""
Integration script to add scraping functionality to the API.

This can be used to add automatic scraping endpoints or scheduled tasks.
"""

from utils.elexys_scraper import ElexysElectricityScraper
from typing import Dict, Any
import sys
import os

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ScrapingService:
    """
    Service class for integrating scraping functionality with the API.

    This can be used to add scraping endpoints to the FastAPI application.
    """

    def __init__(self, db_path: str = None):
        """Initialize the scraping service."""
        self.scraper = ElexysElectricityScraper(db_path=db_path)

    def scrape_latest_prices(self) -> Dict[str, Any]:
        """
        Scrape the latest prices and return results.

        Returns:
            Dictionary with scraping results and statistics
        """
        try:
            # Get current database status
            latest_timestamp = self.scraper.get_latest_db_timestamp()

            # Run scraping
            stats = self.scraper.scrape_and_update()

            # Add timestamp information
            if latest_timestamp:
                stats['previous_latest'] = latest_timestamp.isoformat()

            # Get new latest timestamp
            new_latest = self.scraper.get_latest_db_timestamp()
            if new_latest:
                stats['current_latest'] = new_latest.isoformat()

            return {
                'success': True,
                'statistics': stats,
                'message': f"Scraping completed: {stats.get('inserted', 0)} new records added"
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Scraping failed: {e}"
            }

    def get_scraping_status(self) -> Dict[str, Any]:
        """
        Get the current status of the database and scraping system.

        Returns:
            Dictionary with status information
        """
        try:
            latest_timestamp = self.scraper.get_latest_db_timestamp()

            if latest_timestamp:
                from datetime import datetime
                now = datetime.now()
                age_hours = (now - latest_timestamp).total_seconds() / 3600

                return {
                    'has_data': True,
                    'latest_timestamp': latest_timestamp.isoformat(),
                    'data_age_hours': round(age_hours, 2),
                    'status': 'recent' if age_hours < 24 else 'stale',
                    'message': f"Latest data from {latest_timestamp}"
                }
            else:
                return {
                    'has_data': False,
                    'latest_timestamp': None,
                    'data_age_hours': None,
                    'status': 'empty',
                    'message': "No data in database"
                }

        except Exception as e:
            return {
                'has_data': False,
                'error': str(e),
                'status': 'error',
                'message': f"Error checking status: {e}"
            }


# Example usage functions that could be added to controllers
def create_scraping_endpoints():
    """
    Example of how to add scraping endpoints to FastAPI controllers.

    This function shows the pattern for adding these endpoints to your API.
    """

    # Example endpoint for manual scraping trigger
    example_scrape_endpoint = """
    @router.post(
        "/scrape",
        tags=["Data Management"],
        summary="Manually trigger price data scraping",
        description=\"\"\"
        Manually trigger scraping of the latest electricity price data from Elexys.
        
        This endpoint will:
        - Fetch the latest price data from the Elexys website
        - Check for existing records to avoid duplicates
        - Add only new records to the database
        - Return statistics about the scraping operation
        
        **Use Cases:**
        - Manual data updates when needed
        - Testing the scraping functionality
        - Ensuring database is up to date before analysis
        \"\"\"
    )
    async def trigger_scraping():
        scraping_service = ScrapingService()
        result = scraping_service.scrape_latest_prices()
        
        if result['success']:
            return result
        else:
            raise HTTPException(status_code=500, detail=result['message'])
    """

    # Example endpoint for checking scraping status
    example_status_endpoint = """
    @router.get(
        "/scraping-status",
        tags=["Data Management"],
        summary="Get scraping system status",
        description=\"\"\"
        Get the current status of the price data scraping system.
        
        Returns information about:
        - Latest data timestamp in the database
        - Age of the most recent data
        - Overall system status
        - Recommendations for data updates
        \"\"\"
    )
    async def get_scraping_status():
        scraping_service = ScrapingService()
        return scraping_service.get_scraping_status()
    """

    return {
        'scrape_endpoint': example_scrape_endpoint,
        'status_endpoint': example_status_endpoint
    }


if __name__ == "__main__":
    # Test the service
    service = ScrapingService()

    print("=== Testing Scraping Service ===")

    # Test status check
    status = service.get_scraping_status()
    print(f"Database Status: {status}")

    # Test scraping (commented out to avoid unnecessary scraping)
    # result = service.scrape_latest_prices()
    # print(f"Scraping Result: {result}")

    print("\n=== Example API Integration ===")
    examples = create_scraping_endpoints()
    print("Example endpoints created for FastAPI integration")
    print("Add these to your controllers to enable scraping via API")
