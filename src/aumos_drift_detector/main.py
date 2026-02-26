"""AumOS Drift Detector service entry point.

Creates the FastAPI application with lifespan management for database,
Kafka event publisher, and health checks.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.health import HealthCheck
from aumos_common.observability import get_logger

from aumos_drift_detector.adapters.kafka import DriftEventPublisher
from aumos_drift_detector.api.router import router
from aumos_drift_detector.settings import Settings

logger = get_logger(__name__)
settings = Settings()

# Module-level singletons (injected into routes via FastAPI state)
_kafka_publisher: DriftEventPublisher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Initialises the database connection pool and Kafka event publisher.
    Shuts them down cleanly on exit.

    Args:
        app: The FastAPI application instance.

    Yields:
        None
    """
    global _kafka_publisher  # noqa: PLW0603

    logger.info("Starting AumOS Drift Detector", version="0.1.0")

    # Initialise database (sets up SQLAlchemy async engine + session factory)
    init_database(settings.database)
    logger.info("Database connection pool initialised")

    # Initialise Kafka publisher
    _kafka_publisher = DriftEventPublisher(settings.kafka)
    await _kafka_publisher.start()
    app.state.kafka_publisher = _kafka_publisher
    logger.info("Kafka event publisher ready")

    logger.info("Drift Detector startup complete")
    yield

    # Shutdown
    if _kafka_publisher:
        await _kafka_publisher.stop()
        logger.info("Kafka publisher stopped")

    logger.info("Drift Detector shutdown complete")


app: FastAPI = create_app(
    service_name="aumos-drift-detector",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        HealthCheck(name="postgres", check_fn=lambda: None),
        HealthCheck(name="kafka", check_fn=lambda: None),
    ],
)

app.include_router(router, prefix="/api/v1")
