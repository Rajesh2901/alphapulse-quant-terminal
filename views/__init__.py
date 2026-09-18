"""
AlphaPulse Presentation Views Package
Decoupled UI components and workspace tab renderers.
"""

from views.ai_desk_tab import render_ai_workspace
from views.diagnostics_tab import render_diagnostics_workspace
from views.header_view import render_kpi_cards, render_telemetry_header
from views.risk_tab import render_risk_workspace
from views.terminal_tab import render_terminal_workspace

__all__ = [
    "render_telemetry_header",
    "render_kpi_cards",
    "render_terminal_workspace",
    "render_risk_workspace",
    "render_ai_workspace",
    "render_diagnostics_workspace",
]
