"""
Main FastAPI application with proper MVC architecture.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .controllers import electricity_controller
from .config import app_config


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Initialize FastAPI app with configuration
    app = FastAPI(
        title=app_config.api.title,
        description=app_config.api.description,
        version=app_config.api.version,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Add CORS middleware with configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.api.allow_origins,
        allow_credentials=app_config.api.allow_credentials,
        allow_methods=app_config.api.allow_methods,
        allow_headers=app_config.api.allow_headers,
    )

    # Include routers
    app.include_router(
        electricity_controller.router,
        tags=["Electricity Prices"]
    )

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=app_config.api.host,
        port=app_config.api.port,
        reload=app_config.api.reload
    )
