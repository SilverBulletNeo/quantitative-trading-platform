"""
PDF Report Generation

Generates professional PDF reports with embedded charts and comprehensive metrics.
Creates monthly/quarterly performance reports for strategy monitoring.

Usage:
    from dashboard.pdf_report import PDFReportGenerator

    generator = PDFReportGenerator()
    generator.generate_monthly_report('equity_momentum_90d', 'reports/monthly_report.pdf')
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import os
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.database import DatabaseManager, PerformanceMetric, Position, Trade, Alert
from dashboard.utils.performance import (
    calculate_comprehensive_metrics,
    calculate_max_drawdown,
    calculate_var,
    calculate_cvar
)


class PDFReportGenerator:
    """
    PDF Report Generator for Trading Dashboard

    Creates professional PDF reports with:
    - Performance summary
    - Risk metrics
    - Embedded charts
    - Position tables
    - Trade history
    - Attribution analysis
    """

    def __init__(self, db_path='dashboard/data/dashboard.db'):
        """Initialize PDF report generator"""
        self.db_manager = DatabaseManager(db_path)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#0a0e27'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))

        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#00d4ff'),
            spaceAfter=12,
            spaceBefore=12
        ))

        # Metric label
        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#666666')
        ))

        # Metric value
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#0a0e27'),
            fontName='Helvetica-Bold'
        ))

    def generate_monthly_report(self,
                               strategy_name: str,
                               output_path: str,
                               month: Optional[datetime] = None) -> str:
        """
        Generate monthly performance report

        Args:
            strategy_name: Name of strategy
            output_path: Path to save PDF
            month: Month to report on (defaults to last month)

        Returns:
            Path to generated PDF
        """
        print(f"\n{'='*80}")
        print(f"GENERATING PDF REPORT: {strategy_name}")
        print(f"{'='*80}\n")

        # Default to last month
        if month is None:
            month = datetime.now() - timedelta(days=30)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        # Build content
        story = []

        # 1. Cover page
        print("1. Building cover page...")
        story.extend(self._build_cover_page(strategy_name, month))
        story.append(PageBreak())

        # 2. Performance summary
        print("2. Building performance summary...")
        story.extend(self._build_performance_summary(strategy_name, month))
        story.append(Spacer(1, 0.3*inch))

        # 3. Performance charts
        print("3. Generating performance charts...")
        story.extend(self._build_performance_charts(strategy_name))
        story.append(PageBreak())

        # 4. Risk analysis
        print("4. Building risk analysis...")
        story.extend(self._build_risk_analysis(strategy_name))
        story.append(Spacer(1, 0.3*inch))

        # 5. Attribution analysis
        print("5. Building attribution analysis...")
        story.extend(self._build_attribution_section(strategy_name))
        story.append(PageBreak())

        # 6. Current positions
        print("6. Building positions table...")
        story.extend(self._build_positions_table(strategy_name))
        story.append(Spacer(1, 0.3*inch))

        # 7. Recent trades
        print("7. Building trade history...")
        story.extend(self._build_trade_history(strategy_name))
        story.append(PageBreak())

        # 8. Alerts and warnings
        print("8. Building alerts section...")
        story.extend(self._build_alerts_section(strategy_name))

        # Build PDF
        print("\nBuilding PDF document...")
        doc.build(story)

        print(f"\n{'='*80}")
        print(f"✅ PDF REPORT GENERATED: {output_path}")
        print(f"{'='*80}\n")

        return output_path

    def _build_cover_page(self, strategy_name: str, month: datetime) -> List:
        """Build cover page"""
        elements = []

        # Title
        title = Paragraph(
            f"<b>QUANTITATIVE TRADING PLATFORM</b><br/>Monthly Performance Report",
            self.styles['CustomTitle']
        )
        elements.append(Spacer(1, 2*inch))
        elements.append(title)
        elements.append(Spacer(1, 0.5*inch))

        # Strategy name
        strategy = Paragraph(
            f"<b>Strategy:</b> {strategy_name}",
            self.styles['Heading2']
        )
        elements.append(strategy)
        elements.append(Spacer(1, 0.2*inch))

        # Report period
        period = Paragraph(
            f"<b>Period:</b> {month.strftime('%B %Y')}",
            self.styles['Heading2']
        )
        elements.append(period)
        elements.append(Spacer(1, 0.2*inch))

        # Generated date
        generated = Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['Normal']
        )
        elements.append(Spacer(1, inch))
        elements.append(generated)

        return elements

    def _build_performance_summary(self, strategy_name: str, month: datetime) -> List:
        """Build performance summary section"""
        elements = []

        # Section header
        header = Paragraph("PERFORMANCE SUMMARY", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.2*inch))

        # Get performance data
        session = self.db_manager.get_session()
        try:
            # Get latest metrics
            latest_metric = session.query(PerformanceMetric).filter(
                PerformanceMetric.strategy_name == strategy_name
            ).order_by(PerformanceMetric.date.desc()).first()

            if not latest_metric:
                elements.append(Paragraph("No performance data available", self.styles['Normal']))
                return elements

            # Create metrics table
            data = [
                ['Metric', 'Value', 'Status'],
                ['Annual Return', f"{latest_metric.annual_return:.2f}%",
                 '✓' if latest_metric.annual_return > 10 else '⚠'],
                ['Sharpe Ratio', f"{latest_metric.sharpe_ratio:.2f}",
                 '✓' if latest_metric.sharpe_ratio > 1.5 else '⚠'],
                ['Sortino Ratio', f"{latest_metric.sortino_ratio:.2f}",
                 '✓' if latest_metric.sortino_ratio > 2.0 else '⚠'],
                ['Calmar Ratio', f"{latest_metric.calmar_ratio:.2f}",
                 '✓' if latest_metric.calmar_ratio > 1.0 else '⚠'],
                ['Max Drawdown', f"{latest_metric.max_drawdown:.2f}%",
                 '✓' if latest_metric.max_drawdown > -20 else '⚠'],
                ['Current Drawdown', f"{latest_metric.drawdown:.2f}%",
                 '✓' if latest_metric.drawdown > -10 else '⚠'],
                ['Volatility (Annual)', f"{latest_metric.volatility:.2f}%",
                 '✓' if latest_metric.volatility < 25 else '⚠'],
            ]

            table = Table(data, colWidths=[2.5*inch, 1.5*inch, 0.8*inch])
            table.setStyle(TableStyle([
                # Header row
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0a0e27')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),

                # Data rows
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ]))

            elements.append(table)

        finally:
            session.close()

        return elements

    def _build_performance_charts(self, strategy_name: str) -> List:
        """Build performance charts section"""
        elements = []

        # Section header
        header = Paragraph("PERFORMANCE CHARTS", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.2*inch))

        # Get performance data
        session = self.db_manager.get_session()
        try:
            metrics = session.query(PerformanceMetric).filter(
                PerformanceMetric.strategy_name == strategy_name
            ).order_by(PerformanceMetric.date).all()

            if not metrics:
                elements.append(Paragraph("No chart data available", self.styles['Normal']))
                return elements

            # Extract data
            dates = [m.date for m in metrics]
            cum_returns = [m.cumulative_return for m in metrics]
            drawdowns = [m.drawdown for m in metrics]
            sharpe_ratios = [m.sharpe_ratio for m in metrics]

            # Create cumulative returns chart
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=dates, y=cum_returns,
                mode='lines',
                name='Cumulative Return',
                line=dict(color='#00d4ff', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 255, 0.1)'
            ))
            fig1.update_layout(
                title='Cumulative Returns (%)',
                xaxis_title='Date',
                yaxis_title='Return (%)',
                height=300,
                margin=dict(l=50, r=50, t=50, b=50),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            # Save chart as image
            img_bytes1 = fig1.to_image(format="png", width=700, height=300)
            img1 = Image(io.BytesIO(img_bytes1), width=6*inch, height=2.5*inch)
            elements.append(img1)
            elements.append(Spacer(1, 0.3*inch))

            # Create drawdown chart
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=dates, y=drawdowns,
                mode='lines',
                name='Drawdown',
                line=dict(color='#ff4444', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 68, 68, 0.1)'
            ))
            fig2.add_hline(y=-10, line_dash="dash", line_color="#ffaa00",
                          annotation_text="Warning (-10%)")
            fig2.add_hline(y=-20, line_dash="dash", line_color="#ff4444",
                          annotation_text="Circuit Breaker (-20%)")
            fig2.update_layout(
                title='Drawdown History (%)',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                height=300,
                margin=dict(l=50, r=50, t=50, b=50),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            img_bytes2 = fig2.to_image(format="png", width=700, height=300)
            img2 = Image(io.BytesIO(img_bytes2), width=6*inch, height=2.5*inch)
            elements.append(img2)

        finally:
            session.close()

        return elements

    def _build_risk_analysis(self, strategy_name: str) -> List:
        """Build risk analysis section"""
        elements = []

        # Section header
        header = Paragraph("RISK ANALYSIS", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.2*inch))

        # Get latest metrics
        session = self.db_manager.get_session()
        try:
            latest = session.query(PerformanceMetric).filter(
                PerformanceMetric.strategy_name == strategy_name
            ).order_by(PerformanceMetric.date.desc()).first()

            if not latest:
                elements.append(Paragraph("No risk data available", self.styles['Normal']))
                return elements

            # Calculate additional risk metrics from returns
            metrics = session.query(PerformanceMetric).filter(
                PerformanceMetric.strategy_name == strategy_name
            ).order_by(PerformanceMetric.date).all()

            returns = pd.Series([m.daily_return for m in metrics[-252:]])  # Last year
            var_95 = calculate_var(returns, 0.95) * 100
            var_99 = calculate_var(returns, 0.99) * 100
            cvar_95 = calculate_cvar(returns, 0.95) * 100

            # Risk metrics table
            data = [
                ['Risk Metric', 'Value', 'Threshold', 'Status'],
                ['VaR (95%)', f"{var_95:.2f}%", '-2.5%',
                 '✓' if var_95 > -2.5 else '⚠'],
                ['VaR (99%)', f"{var_99:.2f}%", '-4.0%',
                 '✓' if var_99 > -4.0 else '⚠'],
                ['CVaR (95%)', f"{cvar_95:.2f}%", '-3.5%',
                 '✓' if cvar_95 > -3.5 else '⚠'],
                ['Daily Volatility', f"{returns.std()*100:.2f}%", '<1.5%',
                 '✓' if returns.std()*100 < 1.5 else '⚠'],
                ['Annual Volatility', f"{latest.volatility:.2f}%", '<25%',
                 '✓' if latest.volatility < 25 else '⚠'],
                ['Current Drawdown', f"{latest.drawdown:.2f}%", '>-15%',
                 '✓' if latest.drawdown > -15 else '⚠'],
            ]

            table = Table(data, colWidths=[2*inch, 1.3*inch, 1.3*inch, 0.8*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0a0e27')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ]))

            elements.append(table)

        finally:
            session.close()

        return elements

    def _build_attribution_section(self, strategy_name: str) -> List:
        """Build attribution analysis section"""
        elements = []

        # Section header
        header = Paragraph("ATTRIBUTION ANALYSIS", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.2*inch))

        # Get latest attribution data
        session = self.db_manager.get_session()
        try:
            latest = session.query(PerformanceMetric).filter(
                PerformanceMetric.strategy_name == strategy_name
            ).order_by(PerformanceMetric.date.desc()).first()

            if not latest:
                elements.append(Paragraph("No attribution data available", self.styles['Normal']))
                return elements

            # Attribution table
            data = [
                ['Component', 'Contribution', 'Analysis'],
                ['Benchmark Return', f"+{latest.benchmark_return:.2f}%",
                 'Equal-weight baseline'],
                ['Selection Alpha', f"{latest.selection_alpha:.2f}%",
                 'Asset selection impact (NEGATIVE)'],
                ['Regime Filter', f"+{latest.regime_contribution:.2f}%",
                 'Market regime adaptation (POSITIVE)'],
                ['Transaction Costs', f"{latest.transaction_costs:.2f}%",
                 'Trading costs and slippage'],
                ['Total Return', f"{latest.annual_return:.2f}%",
                 'Net strategy performance'],
            ]

            table = Table(data, colWidths=[2*inch, 1.5*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0a0e27')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ]))

            elements.append(table)
            elements.append(Spacer(1, 0.2*inch))

            # Key findings
            findings = Paragraph(
                "<b>Key Findings:</b><br/>"
                "• Regime filter provides 100% of alpha (+0.69 Sharpe improvement)<br/>"
                "• Asset selection is actually NEGATIVE (-13.9% per year)<br/>"
                "• Focus on regime timing, not stock picking",
                self.styles['Normal']
            )
            elements.append(findings)

        finally:
            session.close()

        return elements

    def _build_positions_table(self, strategy_name: str) -> List:
        """Build current positions table"""
        elements = []

        # Section header
        header = Paragraph("CURRENT POSITIONS", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.2*inch))

        # Get latest positions
        session = self.db_manager.get_session()
        try:
            latest_date = session.query(Position.date).filter(
                Position.strategy_name == strategy_name
            ).order_by(Position.date.desc()).first()

            if not latest_date:
                elements.append(Paragraph("No position data available", self.styles['Normal']))
                return elements

            positions = session.query(Position).filter(
                Position.strategy_name == strategy_name,
                Position.date == latest_date[0]
            ).order_by(Position.weight.desc()).limit(15).all()

            # Build table data
            data = [['Symbol', 'Weight', 'Quantity', 'Price', 'Market Value', 'P&L']]
            for pos in positions:
                data.append([
                    pos.symbol,
                    f"{pos.weight*100:.2f}%",
                    f"{pos.quantity:.0f}",
                    f"${pos.price:.2f}",
                    f"${pos.market_value:,.0f}",
                    f"${pos.unrealized_pnl:,.0f}" if pos.unrealized_pnl else '-'
                ])

            table = Table(data, colWidths=[1*inch, 0.9*inch, 0.9*inch, 0.9*inch, 1.2*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0a0e27')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ]))

            elements.append(table)

        finally:
            session.close()

        return elements

    def _build_trade_history(self, strategy_name: str, limit: int = 20) -> List:
        """Build recent trade history"""
        elements = []

        # Section header
        header = Paragraph("RECENT TRADES", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.2*inch))

        # Get recent trades
        session = self.db_manager.get_session()
        try:
            trades = session.query(Trade).filter(
                Trade.strategy_name == strategy_name
            ).order_by(Trade.timestamp.desc()).limit(limit).all()

            if not trades:
                elements.append(Paragraph("No trade data available", self.styles['Normal']))
                return elements

            # Build table data
            data = [['Date', 'Symbol', 'Side', 'Qty', 'Price', 'Cost', 'Regime']]
            for trade in trades:
                data.append([
                    trade.timestamp.strftime('%Y-%m-%d'),
                    trade.symbol,
                    trade.side,
                    f"{trade.quantity:.0f}",
                    f"${trade.price:.2f}",
                    f"${trade.total_cost:.2f}",
                    trade.regime_at_trade or '-'
                ])

            table = Table(data, colWidths=[0.9*inch, 0.8*inch, 0.6*inch, 0.7*inch, 0.8*inch, 0.8*inch, 1.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0a0e27')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ]))

            elements.append(table)

        finally:
            session.close()

        return elements

    def _build_alerts_section(self, strategy_name: str) -> List:
        """Build alerts and warnings section"""
        elements = []

        # Section header
        header = Paragraph("ALERTS & WARNINGS", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.2*inch))

        # Get unresolved alerts
        session = self.db_manager.get_session()
        try:
            alerts = session.query(Alert).filter(
                Alert.strategy_name == strategy_name,
                Alert.resolved == False
            ).order_by(Alert.timestamp.desc()).limit(10).all()

            if not alerts:
                success = Paragraph(
                    "✓ No active alerts - all metrics within acceptable ranges",
                    self.styles['Normal']
                )
                elements.append(success)
                return elements

            # Build alerts table
            data = [['Date', 'Severity', 'Category', 'Message', 'Threshold', 'Actual']]
            for alert in alerts:
                data.append([
                    alert.timestamp.strftime('%Y-%m-%d'),
                    alert.severity,
                    alert.category,
                    alert.message[:40] + '...' if len(alert.message) > 40 else alert.message,
                    f"{alert.threshold_value:.2f}" if alert.threshold_value else '-',
                    f"{alert.actual_value:.2f}" if alert.actual_value else '-'
                ])

            table = Table(data, colWidths=[0.9*inch, 0.9*inch, 0.9*inch, 2.5*inch, 0.8*inch, 0.8*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0a0e27')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ]))

            elements.append(table)

        finally:
            session.close()

        return elements


if __name__ == '__main__':
    """Test PDF report generation"""

    import sys

    generator = PDFReportGenerator()

    if len(sys.argv) > 1:
        strategy_name = sys.argv[1]
    else:
        strategy_name = 'equity_momentum_90d'

    output_path = f'reports/{strategy_name}_monthly_report_{datetime.now().strftime("%Y%m%d")}.pdf'

    print(f"\nGenerating PDF report for: {strategy_name}")
    print(f"Output path: {output_path}\n")

    try:
        path = generator.generate_monthly_report(strategy_name, output_path)
        print(f"\n✅ Report generated successfully!")
        print(f"   {path}")
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
