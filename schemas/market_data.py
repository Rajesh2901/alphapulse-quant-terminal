"""
AlphaPulse Domain Schemas - Market Data
Strongly-typed Pydantic domain models for asset quotes, ticks, and L2 microstructures.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AssetMetadata(BaseModel):
    """Metadata specification for universe instruments."""

    symbol: str = Field(..., description="Ticker symbol (e.g. NVDA, SPY, BTC-USD)")
    name: str = Field(..., description="Full descriptive asset name")
    category: str = Field(..., description="Asset class or industry sector")

    @field_validator("symbol")
    @classmethod
    def sanitize_symbol(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned or len(cleaned) > 15:
            raise ValueError("Invalid ticker symbol length")
        return cleaned


class MarketQuote(BaseModel):
    """Real-time market quote telemetry."""

    symbol: str
    price: float = Field(..., gt=0, description="Latest traded mid or execution price")
    prev_close: float = Field(..., gt=0, description="Previous session closing price")
    change: float = Field(..., description="Absolute point change from previous close")
    pct_change: float = Field(..., description="Percentage return change from previous close")
    timestamp: float = Field(..., description="Epoch timestamp of telemetry capture")
    status: Literal["LIVE", "SIMULATED", "DELAYED"] = Field(
        default="LIVE", description="Integrity status of the incoming feed"
    )

    @field_validator("price", "prev_close")
    @classmethod
    def round_precision(cls, v: float) -> float:
        return round(v, 4 if v < 1 else 2)


class OrderBookLevel(BaseModel):
    """Individual price level in the Level-2 order ladder."""

    price: float = Field(..., gt=0)
    size: float = Field(..., ge=0)
    total: float = Field(..., ge=0)
    type: Literal["Bid", "Ask"]


class OrderBookState(BaseModel):
    """Aggregate Level-2 market depth book snapshot."""

    spread: float = Field(..., ge=0, description="Absolute bid-ask spread")
    spread_bps: float = Field(..., ge=0, description="Spread in basis points (bps)")
    imbalance: float = Field(
        ..., ge=-1.0, le=1.0, description="Order flow volume imbalance (-1 to +1)"
    )
    best_bid: float = Field(..., gt=0)
    best_ask: float = Field(..., gt=0)
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
