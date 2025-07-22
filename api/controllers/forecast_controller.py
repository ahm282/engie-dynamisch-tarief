from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from ..services.prophet_service import ProphetForecastService
from ..models.forecast_models import ForecastResponse
from ..repositories.prophet_repository import ProphetRepository
from ..weather.weather_collector import WeatherCollector
from ..services.price_service import PriceService
from ..repositories.electricity_price_repository import ElectricityPriceRepository


class ForecastController:
    """Handles electricity price forecasting with comprehensive insights."""

    def __init__(self):
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        @self.router.get("/forecast")
        def get_comprehensive_forecast(
            hours_ahead: int = Query(
                48,
                description="Number of hours to forecast into the future",
                ge=1,
                le=168  # Max 7 days
            ),
            use_ml: bool = Query(
                True,
                description="Use enhanced ML ensemble (recommended for better accuracy)"
            ),
            include_analysis: bool = Query(
                True,
                description="Include weather impact and optimization analysis"
            )
        ):
            """
            **🚀 Complete Electricity Price Forecasting & Analysis**

            This unified endpoint provides everything you need for intelligent energy planning:

            **📊 Core Features:**
            - **Smart Forecasting**: Automatically chooses best model (Prophet + ML ensemble or ML-first) based on data availability
            - **Weather Integration**: Real-time weather conditions + forecast impact on prices  
            - **Price Optimization**: Identifies cheapest periods for energy usage
            - **Solar Intelligence**: Enhanced modeling of solar oversupply effects (1-2 c€/kWh periods)
            - **Actionable Insights**: Specific recommendations for energy management

            **🎯 Perfect For:**
            - EV charging optimization
            - Heat pump scheduling  
            - Solar battery management
            - Industrial energy planning
            - Smart home automation

            **🧠 Adaptive Intelligence:**
            - **Sufficient Data**: Prophet + XGBoost/LightGBM ensemble for maximum accuracy
            - **Limited Data**: ML-first approach with enhanced solar interactions
            - **Weather-Aware**: Incorporates cloud cover, temperature, and solar production factors

            Args:
                hours_ahead: Forecast horizon (1-168 hours, default: 48)
                use_ml: Enable ML enhancement (recommended, default: True)  
                include_analysis: Include weather & optimization analysis (default: True)

            Returns:
                Complete forecasting package with predictions, weather data, optimal periods, and recommendations
            """
            try:
                # Initialize services
                forecast_service = ProphetForecastService()
                weather_collector = WeatherCollector()
                price_repository = ElectricityPriceRepository()

                # 🔮 Generate price forecasts with adaptive strategy
                if use_ml:
                    price_forecasts = forecast_service.ensemble_forecast(
                        hours_ahead=hours_ahead,
                        use_xgboost=True,
                        use_lightgbm=False
                    )
                else:
                    price_forecasts = forecast_service.forecast(hours_ahead)

                # 📊 Basic response structure
                response = {
                    'forecast': {
                        'hours_ahead': hours_ahead,
                        'total_predictions': len(price_forecasts),
                        'predictions': price_forecasts,
                        'model_used': 'ensemble_ml' if use_ml else 'prophet_only',
                        'price_range': self._calculate_price_range(price_forecasts)
                    }
                }

                # 🌤️ Add comprehensive analysis if requested
                if include_analysis:
                    # Current weather conditions
                    current_weather = weather_collector.get_current_weather()

                    # Weather forecast
                    weather_forecast = weather_collector.get_forecast(
                        hours=hours_ahead)

                    # Today's actual prices
                    today_prices = self._get_today_prices(price_repository)

                    # Off-peak analysis for last 7 days
                    offpeak_analysis = forecast_service.get_offpeak_accuracy_analysis(
                        7)

                    # Find optimal energy usage periods
                    optimal_periods = self._find_optimal_periods(
                        price_forecasts, weather_forecast)

                    # Generate smart recommendations
                    recommendations = self._generate_recommendations(
                        current_weather, optimal_periods, price_forecasts, today_prices
                    )

                    # Add analysis to response
                    response.update({
                        'current_conditions': {
                            'weather': current_weather,
                            'today_prices': today_prices
                        },
                        'weather_forecast': {
                            'hours_available': len(weather_forecast) if not weather_forecast.empty else 0,
                            'avg_solar_factor': round(weather_forecast['solar_factor'].mean(), 3) if not weather_forecast.empty else None,
                            'avg_temperature': round(weather_forecast['temperature'].mean(), 1) if not weather_forecast.empty else None
                        },
                        'optimization': {
                            # Top 5 cheapest periods
                            'best_periods': optimal_periods[:5],
                            'total_optimal_periods': len(optimal_periods),
                            'potential_savings': self._calculate_savings(optimal_periods, price_forecasts)
                        },
                        'insights': {
                            'off_peak_analysis': offpeak_analysis,
                            'recommendations': recommendations
                        }
                    })

                return response

            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Forecast error: {str(e)}")

    def _calculate_price_range(self, forecasts):
        """Calculate price statistics"""
        if not forecasts:
            return None

        prices = [f['predicted_price_cents_kwh'] for f in forecasts]
        return {
            'min': round(min(prices), 3),
            'max': round(max(prices), 3),
            'avg': round(sum(prices) / len(prices), 3),
            'median': round(sorted(prices)[len(prices)//2], 3)
        }

    def _get_today_prices(self, repository):
        """Get today's actual electricity prices"""
        try:
            from datetime import datetime
            today = datetime.now().strftime('%Y-%m-%d')
            today_df = repository.find_by_date(today)

            if today_df.empty:
                return {"status": "no_data", "count": 0}

            prices = []
            for _, row in today_df.iterrows():
                prices.append({
                    "timestamp": row['timestamp'],
                    "hour": int(row['hour']),
                    "price_cents_kwh": round(float(row['consumer_price_cents_kwh']), 3)
                })

            return {
                "status": "available",
                "count": len(prices),
                "date": today,
                "prices": prices,
                "avg_price": round(sum(p['price_cents_kwh'] for p in prices) / len(prices), 3)
            }
        except Exception:
            return {"status": "error", "count": 0}

    def _find_optimal_periods(self, forecasts, weather_forecast):
        """Find the best periods for energy usage"""
        optimal_periods = []

        for forecast in forecasts:
            price = forecast.get('predicted_price_cents_kwh', 0)
            period_type = forecast.get('period_type', '')

            # Define optimal criteria
            is_very_cheap = price < 5.0  # Extremely low prices
            is_offpeak_cheap = period_type == 'off-peak' and price < 8.0

            if is_very_cheap or is_offpeak_cheap:
                optimal_periods.append({
                    'timestamp': forecast.get('timestamp'),
                    'price_cents_kwh': price,
                    'period_type': period_type,
                    'savings_opportunity': 'extreme' if is_very_cheap else 'good',
                    'confidence': forecast.get('confidence', 0.8)
                })

        # Sort by price (cheapest first)
        optimal_periods.sort(key=lambda x: x['price_cents_kwh'])
        return optimal_periods

    def _calculate_savings(self, optimal_periods, all_forecasts):
        """Calculate potential savings using optimal periods"""
        if not optimal_periods or not all_forecasts:
            return None

        avg_price = sum(f['predicted_price_cents_kwh']
                        for f in all_forecasts) / len(all_forecasts)
        avg_optimal_price = sum(
            p['price_cents_kwh'] for p in optimal_periods[:5]) / min(5, len(optimal_periods))

        savings_per_kwh = max(0, avg_price - avg_optimal_price)
        savings_percentage = round(
            (savings_per_kwh / avg_price) * 100, 1) if avg_price > 0 else 0

        return {
            'avg_market_price': round(avg_price, 3),
            'avg_optimal_price': round(avg_optimal_price, 3),
            'savings_per_kwh': round(savings_per_kwh, 3),
            'savings_percentage': savings_percentage
        }

    def _generate_recommendations(self, current_weather, optimal_periods, forecasts, today_prices):
        """Generate actionable energy management recommendations"""
        recommendations = []

        # Solar recommendations
        if current_weather:
            solar_factor = current_weather.get('solar_factor', 0)
            if solar_factor > 0.7:
                recommendations.append({
                    'type': 'solar',
                    'priority': 'high',
                    'message': 'Excellent solar conditions - prioritize using solar energy or reduce grid consumption',
                    'action': 'Use stored solar energy, avoid grid usage'
                })
            elif solar_factor < 0.3:
                recommendations.append({
                    'type': 'solar',
                    'priority': 'medium',
                    'message': 'Low solar production expected - rely on grid during optimal periods',
                    'action': 'Plan energy usage during cheap grid periods'
                })

        # Price optimization recommendations
        if optimal_periods:
            next_optimal = optimal_periods[0]
            recommendations.append({
                'type': 'pricing',
                'priority': 'high',
                'message': f"Next cheapest period: {next_optimal['timestamp']} at {next_optimal['price_cents_kwh']:.2f} c€/kWh",
                'action': 'Schedule high-energy activities (EV charging, heat pump, etc.)'
            })

            if len(optimal_periods) >= 3:
                recommendations.append({
                    'type': 'planning',
                    'priority': 'medium',
                    'message': f"Found {len(optimal_periods)} cheap periods - plan energy-intensive tasks",
                    'action': 'Optimize weekly energy schedule around these periods'
                })

        # Temperature-based recommendations
        if current_weather:
            temp = current_weather.get('temperature', 20)
            if temp > 25:
                recommendations.append({
                    'type': 'cooling',
                    'priority': 'medium',
                    'message': 'High temperature - consider pre-cooling during cheap periods',
                    'action': 'Lower thermostat during optimal price periods'
                })
            elif temp < 10:
                recommendations.append({
                    'type': 'heating',
                    'priority': 'medium',
                    'message': 'Low temperature - plan heating during optimal periods',
                    'action': 'Pre-heat home during cheap electricity periods'
                })

        # General efficiency recommendations
        price_volatility = self._calculate_volatility(forecasts)
        if price_volatility > 5:
            recommendations.append({
                'type': 'volatility',
                'priority': 'medium',
                'message': 'High price volatility detected - timing energy usage is crucial',
                'action': 'Avoid energy usage during peak price periods'
            })

        return recommendations

    def _calculate_volatility(self, forecasts):
        """Calculate price volatility"""
        if len(forecasts) < 2:
            return 0

        prices = [f['predicted_price_cents_kwh'] for f in forecasts]
        avg_price = sum(prices) / len(prices)
        variance = sum((p - avg_price) ** 2 for p in prices) / len(prices)
        return variance ** 0.5
