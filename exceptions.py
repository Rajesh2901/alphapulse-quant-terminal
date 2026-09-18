"""
AlphaPulse Enterprise Exceptions
Provides domain-specific exception hierarchies for graceful error boundaries and logging.
"""


class AlphaPulseBaseException(Exception):
    """Base exception for all AlphaPulse domain errors."""

    def __init__(self, message: str, details: str = ""):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class MarketDataError(AlphaPulseBaseException):
    """Raised when ingestion from market data endpoints fails or returns corrupt data."""

    pass


class InsufficientDataError(AlphaPulseBaseException):
    """Raised when historical bars are fewer than required for statistical validity."""

    pass


class ModelConvergenceError(AlphaPulseBaseException):
    """Raised when econometric (ARIMA) or machine learning optimization fails to converge."""

    pass


class InvalidSymbolError(AlphaPulseBaseException):
    """Raised when an unparseable or malicious ticker symbol input is supplied."""

    pass


class AIAnalystError(AlphaPulseBaseException):
    """Raised when Google Gemini API connectivity or parsing fails."""

    pass


class ConfigurationError(AlphaPulseBaseException):
    """Raised when critical configuration tokens or model parameters are invalid."""

    pass
