"""
Web scraper for Elexys electricity price data with weather integration.

This module scrapes electricity price data from the Elexys Spot Belpex page,
processes the data, and stores it in the database while avoiding duplicates.
Enhanced with weather data collection for improved forecasting accuracy.
"""

from api.weather.weather_collector import WeatherCollector
from api.config.database import DatabaseManager
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple
import sys
import os

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# Import weather collector for historical weather data


class ElexysElectricityScraper:
    """
    Scraper for Elexys electricity price data with weather integration.

    Features:
    - Scrapes data from Elexys Spot Belpex table
    - Processes date/time and price information
    - Collects weather data (cloud cover, temperature, solar factor) for each timestamp
    - Checks for existing data to avoid duplicates
    - Updates existing records with missing weather data
    - Calculates consumer prices with markup
    - Robust error handling and logging
    """

    def __init__(self, db_path: str = None):
        """
        Initialize the scraper.

        Args:
            db_path: Optional path to the database file
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Database setup
        if db_path:
            self.db_path = db_path
        else:
            # Calculate absolute path more reliably
            # This file is in: api/utils/scraping/elexys_scraper.py
            # Database is at: api/db/electricity_prices.db
            current_file = os.path.abspath(__file__)
            utils_dir = os.path.dirname(current_file)  # api/utils/scraping
            api_dir = os.path.dirname(os.path.dirname(utils_dir))  # api
            self.db_path = os.path.join(api_dir, 'db', 'electricity_prices.db')

        # Ensure database directory exists
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            self.logger.info(f"Created database directory: {db_dir}")

        self.logger.info(f"Using database path: {self.db_path}")

        self.db_manager = DatabaseManager()

        # Initialize weather collector for historical data
        self.weather_collector = WeatherCollector()

        # Scraping configuration
        self.base_url = "https://my.elexys.be/MarketInformation/SpotBelpex.aspx"
        self.table_id = "contentPlaceHolder_belpexFilterGrid_DXMainTable"
        self.row_class_prefix = "contentPlaceHolder_belpexFilterGrid_DXDataRow"

        # Session for persistent connections
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
        })

    def fetch_webpage(self) -> Optional[BeautifulSoup]:
        """
        Fetch the Elexys webpage and parse it.

        Returns:
            BeautifulSoup object or None if failed
        """
        try:
            self.logger.info(f"Fetching webpage: {self.base_url}")
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            self.logger.info("Successfully fetched and parsed webpage")
            return soup

        except requests.RequestException as e:
            self.logger.error(f"Error fetching webpage: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error parsing webpage: {e}")
            return None

    def extract_table_data(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """
        Extract price data from the table.

        Args:
            soup: BeautifulSoup object of the webpage

        Returns:
            List of dictionaries containing the extracted data
        """
        try:
            # Find the main table
            table = soup.find('table', {'id': self.table_id})
            if not table:
                self.logger.error(f"Table with ID '{self.table_id}' not found")
                return []

            # Find all data rows using the correct ID pattern
            data_rows = []
            row_index = 0

            while True:
                row_id = f"{self.row_class_prefix}{row_index}"
                row = soup.find('tr', {'id': row_id})

                if not row:
                    # Try with class instead of ID
                    row = soup.find(
                        'tr', class_=f"{self.row_class_prefix}{row_index}")
                    if not row:
                        break

                # Extract date and price from the row
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    date_cell = cells[0].get_text(strip=True)
                    price_cell = cells[1].get_text(strip=True)

                    # Only add rows that have valid date and price data
                    if date_cell and price_cell and '/' in date_cell and '€' in price_cell:
                        data_rows.append({
                            'date_time': date_cell,
                            'price_raw': price_cell,
                            'row_index': row_index
                        })

                row_index += 1

                # Safety limit to prevent infinite loops
                if row_index > 300:
                    break

            self.logger.info(f"Extracted {len(data_rows)} rows from table")
            return data_rows

        except Exception as e:
            self.logger.error(f"Error extracting table data: {e}")
            return []

    def parse_datetime(self, date_time_str: str) -> Optional[Tuple[datetime, str, int]]:
        """
        Parse date and time from the scraped string.

        Args:
            date_time_str: Date/time string like "18/07/2025 23:00:00"

        Returns:
            Tuple of (datetime object, date string, hour) or None if parsing fails
        """
        try:
            # Remove any extra whitespace and normalize
            date_time_str = date_time_str.strip()

            # Parse the datetime string
            # Expected format: "dd/mm/yyyy hh:mm:ss"
            dt = datetime.strptime(date_time_str, "%d/%m/%Y %H:%M:%S")

            # Format date as YYYY-MM-DD for database consistency
            date_str = dt.strftime("%Y-%m-%d")
            hour = dt.hour

            return dt, date_str, hour

        except ValueError as e:
            self.logger.error(f"Error parsing datetime '{date_time_str}': {e}")
            return None

    def parse_price(self, price_str: str) -> Optional[float]:
        """
        Parse price from the scraped string.

        Args:
            price_str: Price string like "€ 113,87"

        Returns:
            Price as float in EUR/MWh or None if parsing fails
        """
        try:
            # Remove currency symbol, spaces, and normalize
            price_str = price_str.replace('€', '').replace(' ', '').strip()

            # Handle European decimal format (comma as decimal separator)
            price_str = price_str.replace(',', '.')

            # Convert to float
            price = float(price_str)

            return price

        except (ValueError, TypeError) as e:
            self.logger.error(f"Error parsing price '{price_str}': {e}")
            return None

    def calculate_consumer_price(self, wholesale_price: float) -> float:
        """
        Calculate consumer price from wholesale price.

        Args:
            wholesale_price: Wholesale price in EUR/MWh

        Returns:
            Consumer price in euro cents per kWh
        """
        # Use the corrected consumer price formula: 1.3163 + (0.1019 * wholesale)
        consumer_price = 1.3163 + (0.1019 * wholesale_price)

        return round(consumer_price, 4)

    def get_weather_data(self, timestamp: datetime) -> Dict[str, Optional[float]]:
        """
        Get weather data for a specific timestamp.

        Args:
            timestamp: The datetime to get weather data for

        Returns:
            Dictionary with weather data (cloud_cover, temperature, solar_factor)
        """
        try:
            # For historical data, use weather proxy since API doesn't provide historical data
            weather_data = self.weather_collector.get_historical_weather_proxy([
                                                                               timestamp])

            if not weather_data.empty:
                # Get the weather record
                weather_record = weather_data.iloc[0]

                return {
                    'cloud_cover': float(weather_record.get('cloud_cover', 50.0)),
                    'temperature': float(weather_record.get('temperature', 15.0)),
                    'solar_factor': float(weather_record.get('solar_factor', 0.5))
                }
            else:
                # Fallback to simple proxy values
                return {
                    'cloud_cover': 50.0,  # Default moderate cloudiness
                    'temperature': 15.0,  # Default moderate temperature
                    'solar_factor': 0.5   # Default moderate solar factor
                }

        except Exception as e:
            self.logger.warning(
                f"Could not get weather data for {timestamp}: {e}")
            # Return default values if weather collection fails
            return {
                'cloud_cover': 50.0,
                'temperature': 15.0,
                'solar_factor': 0.5
            }

    def check_existing_record(self, timestamp: datetime) -> Optional[Dict[str, any]]:
        """
        Check if a record already exists in the database and return its data.

        Args:
            timestamp: The timestamp to check

        Returns:
            Dictionary with existing record data or None if record doesn't exist
        """
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.execute(
                "SELECT price_eur, price_raw, consumer_price_cents_kwh, cloud_cover, temperature, solar_factor FROM electricity_prices WHERE timestamp = ?",
                (timestamp.strftime("%Y-%m-%d %H:%M:%S"),)
            )
            result = cursor.fetchone()
            conn.close()

            if result:
                return {
                    'price_eur': result[0],
                    'price_raw': result[1],
                    'consumer_price_cents_kwh': result[2],
                    'cloud_cover': result[3],
                    'temperature': result[4],
                    'solar_factor': result[5]
                }
            return None

        except Exception as e:
            self.logger.error(f"Error checking existing record: {e}")
            return None

    def update_price_record(self, timestamp: datetime, date_str: str, hour: int,
                            price_eur: float, price_raw: str, consumer_price: float,
                            weather_data: Dict[str, Optional[float]]) -> bool:
        """
        Update an existing price record in the database.

        Args:
            timestamp: Datetime object
            date_str: Date string in YYYY-MM-DD format
            hour: Hour of the day (0-23)
            price_eur: Wholesale price in EUR/MWh
            price_raw: Raw price string from website
            consumer_price: Consumer price in cents/kWh
            weather_data: Dictionary with weather data (cloud_cover, temperature, solar_factor)

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self.db_manager.get_connection()

            cursor = conn.execute("""
                UPDATE electricity_prices 
                SET date = ?, hour = ?, price_eur = ?, price_raw = ?, consumer_price_cents_kwh = ?,
                    cloud_cover = ?, temperature = ?, solar_factor = ?
                WHERE timestamp = ?
            """, (
                date_str,
                hour,
                price_eur,
                price_raw,
                consumer_price,
                float(weather_data.get('cloud_cover')) if weather_data.get(
                    'cloud_cover') is not None else None,
                float(weather_data.get('temperature')) if weather_data.get(
                    'temperature') is not None else None,
                float(weather_data.get('solar_factor')) if weather_data.get(
                    'solar_factor') is not None else None,
                timestamp.strftime("%Y-%m-%d %H:%M:%S")
            ))

            conn.commit()
            conn.close()

            self.logger.info(
                f"Updated record: {timestamp} - {price_eur} EUR/MWh (with weather data)")
            return True

        except Exception as e:
            self.logger.error(f"Error updating record: {e}")
            return False

    def insert_price_record(self, timestamp: datetime, date_str: str, hour: int,
                            price_eur: float, price_raw: str, consumer_price: float,
                            weather_data: Dict[str, Optional[float]]) -> bool:
        """
        Insert a price record into the database.

        Args:
            timestamp: Datetime object
            date_str: Date string in YYYY-MM-DD format
            hour: Hour of the day (0-23)
            price_eur: Wholesale price in EUR/MWh
            price_raw: Raw price string from website
            consumer_price: Consumer price in cents/kWh
            weather_data: Dictionary with weather data (cloud_cover, temperature, solar_factor)

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self.db_manager.get_connection()

            cursor = conn.execute("""
                INSERT INTO electricity_prices 
                (timestamp, date, hour, price_eur, price_raw, consumer_price_cents_kwh,
                 cloud_cover, temperature, solar_factor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                date_str,
                hour,
                price_eur,
                price_raw,
                consumer_price,
                float(weather_data.get('cloud_cover')) if weather_data.get(
                    'cloud_cover') is not None else None,
                float(weather_data.get('temperature')) if weather_data.get(
                    'temperature') is not None else None,
                float(weather_data.get('solar_factor')) if weather_data.get(
                    'solar_factor') is not None else None
            ))

            conn.commit()
            conn.close()

            self.logger.debug(
                f"Inserted record: {timestamp} - {price_eur} EUR/MWh (with weather data)")
            return True

        except Exception as e:
            self.logger.error(f"Error inserting record: {e}")
            return False

    def process_scraped_data(self, data_rows: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Process scraped data and insert/update in database.

        Args:
            data_rows: List of scraped data dictionaries

        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'total_rows': len(data_rows),
            'processed': 0,
            'skipped_existing': 0,
            'skipped_errors': 0,
            'inserted': 0,
            'updated': 0
        }

        # Process rows in reverse order (oldest first)
        for row_data in reversed(data_rows):
            stats['processed'] += 1

            # Parse datetime
            datetime_result = self.parse_datetime(row_data['date_time'])
            if not datetime_result:
                stats['skipped_errors'] += 1
                continue

            timestamp, date_str, hour = datetime_result

            # Parse price
            price_eur = self.parse_price(row_data['price_raw'])
            if price_eur is None:
                stats['skipped_errors'] += 1
                continue

            # Calculate consumer price
            consumer_price = self.calculate_consumer_price(price_eur)

            # Get weather data for this timestamp
            weather_data = self.get_weather_data(timestamp)

            # Check if record already exists
            existing_record = self.check_existing_record(timestamp)

            if existing_record:
                # Compare prices to see if update is needed
                price_diff = abs(existing_record['price_eur'] - price_eur)
                consumer_price_diff = abs(
                    existing_record['consumer_price_cents_kwh'] - consumer_price)

                # Also check if weather data is missing or significantly different
                weather_update_needed = (
                    existing_record.get('cloud_cover') is None or
                    existing_record.get('temperature') is None or
                    existing_record.get('solar_factor') is None
                )

                # Update if there's a significant difference or missing weather data
                if (price_diff > 0.01 or consumer_price_diff > 0.001 or
                    existing_record['price_raw'] != row_data['price_raw'] or
                        weather_update_needed):

                    update_reason = []
                    if price_diff > 0.01:
                        update_reason.append(
                            f"price change ({existing_record['price_eur']:.2f} → {price_eur:.2f})")
                    if weather_update_needed:
                        update_reason.append("missing weather data")
                    if existing_record['price_raw'] != row_data['price_raw']:
                        update_reason.append("raw price format change")

                    self.logger.info(
                        f"Updating record for {timestamp}: {', '.join(update_reason)}"
                    )

                    if self.update_price_record(
                        timestamp, date_str, hour, price_eur,
                        row_data['price_raw'], consumer_price, weather_data
                    ):
                        stats['updated'] += 1
                    else:
                        stats['skipped_errors'] += 1
                else:
                    self.logger.debug(
                        f"Record already exists for {timestamp} with same data, skipping")
                    stats['skipped_existing'] += 1
            else:
                # Insert new record with weather data
                if self.insert_price_record(
                    timestamp, date_str, hour, price_eur,
                    row_data['price_raw'], consumer_price, weather_data
                ):
                    stats['inserted'] += 1
                else:
                    stats['skipped_errors'] += 1

        return stats

    def scrape_and_update(self) -> Dict[str, int]:
        """
        Main method to scrape data and update the database.

        Returns:
            Dictionary with scraping statistics
        """
        self.logger.info("Starting Elexys electricity price scraping")

        try:
            # Fetch webpage
            soup = self.fetch_webpage()
            if not soup:
                return {'error': 'Failed to fetch webpage'}

            # Extract table data
            data_rows = self.extract_table_data(soup)
            if not data_rows:
                return {'error': 'No data extracted from table'}

            # Process and insert data
            stats = self.process_scraped_data(data_rows)

            self.logger.info(f"Scraping completed: {stats}")
            return stats

        except Exception as e:
            self.logger.error(f"Error during scraping: {e}")
            return {'error': str(e)}

    def get_latest_db_timestamp(self) -> Optional[datetime]:
        """
        Get the latest timestamp from the database.

        Returns:
            Latest datetime or None if no data
        """
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.execute(
                "SELECT MAX(timestamp) FROM electricity_prices"
            )
            result = cursor.fetchone()[0]
            conn.close()

            if result:
                return datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
            return None

        except Exception as e:
            self.logger.error(f"Error getting latest timestamp: {e}")
            return None

    def print_stats(self, stats: Dict[str, int]):
        """Print scraping statistics in a readable format."""
        print("\n" + "="*50)
        print("ELEXYS SCRAPING RESULTS (WITH WEATHER DATA)")
        print("="*50)

        if 'error' in stats:
            print(f"❌ Error: {stats['error']}")
            return

        print(f"📊 Total rows found: {stats.get('total_rows', 0)}")
        print(f"🔄 Rows processed: {stats.get('processed', 0)}")
        print(
            f"✅ New records inserted: {stats.get('inserted', 0)} (with weather data)")
        print(
            f"🔄 Records updated: {stats.get('updated', 0)} (including weather data)")
        print(
            f"⏭️  Existing records skipped: {stats.get('skipped_existing', 0)}")
        print(f"❌ Rows with errors: {stats.get('skipped_errors', 0)}")

        if stats.get('inserted', 0) > 0:
            print(
                f"\n🎉 Successfully added {stats['inserted']} new price records with weather data!")

        if stats.get('updated', 0) > 0:
            print(
                f"\n🔄 Successfully updated {stats['updated']} existing records with weather data!")

        if stats.get('inserted', 0) == 0 and stats.get('updated', 0) == 0 and stats.get('skipped_existing', 0) > 0:
            print(f"\n✨ Database is up to date - no new records needed")
        elif stats.get('inserted', 0) == 0 and stats.get('updated', 0) == 0:
            print(f"\n⚠️  No new records were added or updated")

        print("\n🌤️  Weather data is now being collected and stored with all price records")
        print("="*50)


def main():
    """Main function for command-line usage."""
    scraper = ElexysElectricityScraper()

    # Show current database status
    latest_timestamp = scraper.get_latest_db_timestamp()
    if latest_timestamp:
        print(f"📅 Latest data in database: {latest_timestamp}")
    else:
        print("📅 No existing data in database")

    # Run scraping
    stats = scraper.scrape_and_update()
    scraper.print_stats(stats)


if __name__ == "__main__":
    main()
