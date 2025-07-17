# Electricity Price API - Refactored

A Spring Boot-inspired FastAPI application for electricity price data analysis with proper MVC architecture.

## Architecture

This API follows Spring Boot-like patterns with clear separation of concerns:

```
api/
├── config/                 # Configuration management
│   ├── settings.py         # Application settings
│   └── __init__.py
├── controllers/            # HTTP request handlers
│   └── __init__.py         # ElectricityPriceController
├── services/               # Business logic layer
│   └── __init__.py         # ElectricityPriceService
├── repositories/           # Data access layer
│   ├── base_repository.py  # Abstract base repository
│   ├── electricity_price_repository.py
│   └── __init__.py
├── models/                 # Data models
│   ├── price_models.py     # Price-related models
│   ├── stats_models.py     # Statistics models
│   ├── negative_price_models.py
│   ├── response_models.py  # API response models
│   └── __init__.py
├── database/               # Database management
│   └── __init__.py         # DatabaseManager
├── main.py                 # Application factory
└── requirements.txt        # Dependencies
```

## Design Patterns

### 1. Dependency Injection

-   Services are injected into controllers using FastAPI's `Depends()`
-   Repositories are injected into services via constructor
-   Configuration is injected into components

### 2. Repository Pattern

-   `BaseRepository` abstract interface
-   `ElectricityPriceRepository` implements data access
-   Clear separation between business logic and data access

### 3. Service Layer

-   `ElectricityPriceService` contains business logic
-   No static methods - proper OOP with instance methods
-   Services depend on repositories, not direct database access

### 4. Configuration Management

-   `ApplicationConfig` for centralized settings
-   Environment variable support via Pydantic
-   Separate configs for database, API, etc.

### 5. Model Separation

-   Separate files for different model types
-   Clear imports and exports
-   Type safety with Pydantic

## Key Features

-   **MVC Architecture**: Clear separation of Model, View (API), Controller
-   **Dependency Injection**: Loose coupling between components
-   **Configuration Management**: Environment-based configuration
-   **Type Safety**: Full Pydantic model validation
-   **Error Handling**: Comprehensive exception handling
-   **Health Checks**: Built-in health monitoring
-   **API Documentation**: Auto-generated OpenAPI docs

## Usage

### Start the API

```bash
# Using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Using Python module
python -m api.main
```

### Configuration

Set environment variables to override defaults:

```bash
export API_PORT=8000
export API_DEBUG=true
export DB_DATABASE_PATH=custom/path/to/database.db
```

### Endpoints

-   `GET /` - API information
-   `GET /health` - Health check
-   `GET /database-info` - Database statistics
-   `GET /prices` - Price data with filters
-   `GET /current-prices` - Today's price data with categorization
-   `GET /next-day-prices` - Tomorrow's price data (if available)
-   `GET /prices-by-date` - Price data for any specific date
-   `GET /all-prices` - Complete price dataset export
-   `GET /consumption-cost` - Consumption cost analysis
-   `GET /daily-stats` - Daily statistics
-   `GET /hourly-stats` - Hourly patterns
-   `GET /extremes` - Highest/lowest prices
-   `GET /negative-prices` - Negative price analysis
-   `GET /expensive-prices` - Expensive price monitoring
-   `GET /expensive-prices/summary` - Expensive price summary
-   `GET /expensive-prices/top` - Top expensive prices
-   `GET /expensive-prices/trends` - Expensive price trends
-   `GET /expensive-prices/percentiles` - Price percentiles analysis

## Benefits of This Architecture

1. **Maintainability**: Clear separation of concerns
2. **Testability**: Easy to mock dependencies
3. **Scalability**: Easy to add new features
4. **Flexibility**: Configuration-driven behavior
5. **Professional**: Industry-standard patterns
