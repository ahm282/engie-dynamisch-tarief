# Elexys Electricity Price Scraper

This module provides web scraping functionality for automatically fetching electricity price data from the Elexys Spot Belpex website and storing it in the database.

## Features

-   **Automated Data Extraction**: Scrapes electricity price data from Elexys website
-   **Duplicate Prevention**: Checks existing database records to avoid duplicates
-   **Consumer Price Calculation**: Automatically calculates consumer prices with realistic markup
-   **Robust Error Handling**: Comprehensive error handling and logging
-   **Command-Line Interface**: Easy-to-use CLI for manual scraping

## Files Overview

### Core Scraper

-   **`elexys_scraper.py`** - Main scraping class with full functionality
-   **`scrape_prices.py`** - Command-line utility for running the scraper

## Quick Start

### Command Line Usage

**Important**: Run all commands from the project root directory (`e:\Python\Huishouden\Elektriciteit`), not from the utils folder.

```bash
# Basic scraping (most common usage)
python utils/scrape_prices.py

# Check database status without scraping
python utils/scrape_prices.py --check-only

# Verbose output for debugging
python utils/scrape_prices.py --verbose

# Quiet mode (only errors)
python utils/scrape_prices.py --quiet

# Use custom database file
python utils/scrape_prices.py --db-path custom.db
```

### Programmatic Usage

```python
from utils.elexys_scraper import ElexysElectricityScraper

# Create scraper instance
scraper = ElexysElectricityScraper()

# Run scraping and get statistics
stats = scraper.scrape_and_update()
print(f"Added {stats['inserted']} new records")

# Check latest data
latest = scraper.get_latest_db_timestamp()
print(f"Latest data: {latest}")
```

## How It Works

### 1. Website Analysis

The scraper targets the Elexys Spot Belpex page:

-   **URL**: `https://my.elexys.be/MarketInformation/SpotBelpex.aspx`
-   **Table ID**: `contentPlaceHolder_belpexFilterGrid_DXMainTable`
-   **Data Rows**: `contentPlaceHolder_belpexFilterGrid_DXDataRow0` to `contentPlaceHolder_belpexFilterGrid_DXDataRow215`

### 2. Data Processing

1. **Fetch**: Downloads the webpage using requests with proper headers
2. **Parse**: Uses BeautifulSoup to extract table data
3. **Validate**: Checks date/time and price formats
4. **Transform**: Converts European date format and euro prices
5. **Store**: Inserts new records into SQLite database

### 3. Price Calculations

-   **Wholesale Price**: Direct from website (EUR/MWh)
-   **Consumer Price**: Calculated with 25% markup + 3.5 c€/kWh network costs
-   **Price Categories**: Automatically categorized (cheap, regular, expensive, extremely expensive)

### 4. Duplicate Prevention

-   Checks existing database records by timestamp
-   Skips already imported data
-   Processes only new records in reverse chronological order

## Data Format

### Input Data (from website)

```
Date/Time: "18/07/2025 23:00:00"
Price: "€ 113,87"
```

### Database Storage

```sql
timestamp: "2025-07-18 23:00:00"
date: "2025-07-18"
hour: 23
price_eur: 113.87
price_raw: "€ 113,87"
consumer_price_cents_kwh: 17.887
```

## Configuration

### Scraper Settings

```python
# Network configuration
consumer_markup_percentage = 25.0  # 25% markup
network_costs = 3.5  # Additional costs in cents/kWh

# HTTP headers for web requests
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...',
    'Accept': 'text/html,application/xhtml+xml,application/xml...',
    # ... additional headers
}
```

### Database Schema

The scraper works with the existing `electricity_prices` table:

```sql
CREATE TABLE electricity_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    date TEXT NOT NULL,
    hour INTEGER NOT NULL,
    price_eur REAL NOT NULL,
    price_raw TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    consumer_price_cents_kwh REAL
);
```

## Error Handling

The scraper includes comprehensive error handling:

-   **Network Errors**: Timeout, connection issues, HTTP errors
-   **Parsing Errors**: Invalid HTML, missing table elements
-   **Data Errors**: Invalid date formats, unparseable prices
-   **Database Errors**: Connection issues, constraint violations

All errors are logged with detailed information for debugging.

## Logging

Default logging configuration:

-   **INFO Level**: General operation status
-   **DEBUG Level**: Detailed parsing information (use `--verbose`)
-   **ERROR Level**: Only errors (use `--quiet`)

## Typical Output

```
📅 Latest data in database: 2025-07-18 23:00:00
🚀 Starting electricity price scraping...

2025-07-17 14:43:55,582 - INFO - Starting Elexys electricity price scraping
2025-07-17 14:43:55,805 - INFO - Successfully fetched and parsed webpage
2025-07-17 14:43:56,045 - INFO - Extracted 216 rows from table

==================================================
ELEXYS SCRAPING RESULTS
==================================================
📊 Total rows found: 216
🔄 Rows processed: 216
✅ New records inserted: 24
⏭️  Existing records skipped: 192
❌ Rows with errors: 0

🎉 Successfully added 24 new price records!
==================================================
```

## Scheduling

For automated scraping, you can set up scheduled tasks:

### Windows Task Scheduler

```batch
# Run every hour
python "E:\Python\Huishouden\Elektriciteit\utils\scrape_prices.py" --quiet
```

### Cron (Linux/Mac)

```bash
# Run every hour
0 * * * * /usr/bin/python3 /path/to/project/utils/scrape_prices.py --quiet
```

### Python Scheduler

```python
import schedule
import time
from utils.elexys_scraper import ElexysElectricityScraper

def scrape_job():
    scraper = ElexysElectricityScraper()
    stats = scraper.scrape_and_update()
    print(f"Scraping completed: {stats['inserted']} new records")

# Schedule every hour
schedule.every().hour.do(scrape_job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Dependencies

Required Python packages:

-   `requests` - HTTP client for web scraping
-   `beautifulsoup4` - HTML parsing
-   `pandas` - Data processing (existing dependency)
-   `sqlite3` - Database operations (built-in)

Install missing dependencies:

```bash
pip install requests beautifulsoup4
```

## Performance

-   **Scraping Time**: ~2-3 seconds for full update
-   **Data Volume**: ~216 records per scrape (9 days of hourly data)
-   **Memory Usage**: Minimal (processes data incrementally)
-   **Network Usage**: Single HTTP request (~116KB)
