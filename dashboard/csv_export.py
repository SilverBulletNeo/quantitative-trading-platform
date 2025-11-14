"""
CSV Export Functionality

Provides data export capabilities for the dashboard.
Exports returns, positions, trades, and alerts to CSV format.

Usage:
    from dashboard.csv_export import CSVExporter

    exporter = CSVExporter()
    exporter.export_returns('equity_momentum_90d', 'exports/returns.csv')
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import os

from dashboard.database import db_manager, PerformanceMetric, Position, Trade, Alert


class CSVExporter:
    """
    CSV Export Manager

    Handles all data export operations from dashboard database to CSV files.
    """

    def __init__(self, db_path='dashboard/data/dashboard.db'):
        """Initialize CSV exporter"""
        from dashboard.database import DatabaseManager
        self.db_manager = DatabaseManager(db_path)

    def export_returns(self,
                      strategy_name: str,
                      output_path: str,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> str:
        """
        Export daily returns to CSV

        Args:
            strategy_name: Name of strategy
            output_path: Path to save CSV
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Path to exported file
        """
        session = self.db_manager.get_session()

        try:
            # Query performance metrics
            query = session.query(PerformanceMetric).filter(
                PerformanceMetric.strategy_name == strategy_name
            )

            if start_date:
                query = query.filter(PerformanceMetric.date >= start_date)
            if end_date:
                query = query.filter(PerformanceMetric.date <= end_date)

            query = query.order_by(PerformanceMetric.date)

            # Convert to DataFrame
            data = []
            for record in query.all():
                data.append({
                    'date': record.date,
                    'daily_return': record.daily_return,
                    'cumulative_return': record.cumulative_return,
                    'sharpe_ratio': record.sharpe_ratio,
                    'drawdown': record.drawdown,
                    'volatility': record.volatility
                })

            df = pd.DataFrame(data)

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Export to CSV
            df.to_csv(output_path, index=False)

            print(f"✅ Exported {len(df)} daily returns to {output_path}")
            return output_path

        finally:
            session.close()

    def export_positions(self,
                        strategy_name: str,
                        output_path: str,
                        date: Optional[datetime] = None) -> str:
        """
        Export positions to CSV

        Args:
            strategy_name: Name of strategy
            output_path: Path to save CSV
            date: Specific date (defaults to latest)

        Returns:
            Path to exported file
        """
        session = self.db_manager.get_session()

        try:
            # Get latest date if not specified
            if date is None:
                latest = session.query(Position.date).filter(
                    Position.strategy_name == strategy_name
                ).order_by(Position.date.desc()).first()

                if latest:
                    date = latest[0]
                else:
                    print("⚠️  No positions found")
                    return None

            # Query positions
            query = session.query(Position).filter(
                Position.strategy_name == strategy_name,
                Position.date == date
            ).order_by(Position.weight.desc())

            # Convert to DataFrame
            data = []
            for record in query.all():
                data.append({
                    'symbol': record.symbol,
                    'weight': record.weight,
                    'quantity': record.quantity,
                    'price': record.price,
                    'market_value': record.market_value,
                    'unrealized_pnl': record.unrealized_pnl
                })

            df = pd.DataFrame(data)

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Export to CSV
            df.to_csv(output_path, index=False)

            print(f"✅ Exported {len(df)} positions for {date.date()} to {output_path}")
            return output_path

        finally:
            session.close()

    def export_trades(self,
                     strategy_name: str,
                     output_path: str,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     limit: int = 1000) -> str:
        """
        Export trade history to CSV

        Args:
            strategy_name: Name of strategy
            output_path: Path to save CSV
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of trades to export

        Returns:
            Path to exported file
        """
        session = self.db_manager.get_session()

        try:
            # Query trades
            query = session.query(Trade).filter(
                Trade.strategy_name == strategy_name
            )

            if start_date:
                query = query.filter(Trade.timestamp >= start_date)
            if end_date:
                query = query.filter(Trade.timestamp <= end_date)

            query = query.order_by(Trade.timestamp.desc()).limit(limit)

            # Convert to DataFrame
            data = []
            for record in query.all():
                data.append({
                    'timestamp': record.timestamp,
                    'symbol': record.symbol,
                    'side': record.side,
                    'quantity': record.quantity,
                    'price': record.price,
                    'commission': record.commission,
                    'slippage': record.slippage,
                    'total_cost': record.total_cost,
                    'regime': record.regime_at_trade,
                    'reason': record.reason
                })

            df = pd.DataFrame(data)

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Export to CSV
            df.to_csv(output_path, index=False)

            print(f"✅ Exported {len(df)} trades to {output_path}")
            return output_path

        finally:
            session.close()

    def export_alerts(self,
                     strategy_name: Optional[str] = None,
                     output_path: str = 'exports/alerts.csv',
                     include_resolved: bool = False,
                     limit: int = 500) -> str:
        """
        Export alerts to CSV

        Args:
            strategy_name: Optional strategy name filter
            output_path: Path to save CSV
            include_resolved: Whether to include resolved alerts
            limit: Maximum number of alerts to export

        Returns:
            Path to exported file
        """
        session = self.db_manager.get_session()

        try:
            # Query alerts
            query = session.query(Alert)

            if strategy_name:
                query = query.filter(Alert.strategy_name == strategy_name)

            if not include_resolved:
                query = query.filter(Alert.resolved == False)

            query = query.order_by(Alert.timestamp.desc()).limit(limit)

            # Convert to DataFrame
            data = []
            for record in query.all():
                data.append({
                    'timestamp': record.timestamp,
                    'strategy': record.strategy_name,
                    'severity': record.severity,
                    'category': record.category,
                    'message': record.message,
                    'threshold': record.threshold_value,
                    'actual': record.actual_value,
                    'acknowledged': record.acknowledged,
                    'resolved': record.resolved
                })

            df = pd.DataFrame(data)

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Export to CSV
            df.to_csv(output_path, index=False)

            print(f"✅ Exported {len(df)} alerts to {output_path}")
            return output_path

        finally:
            session.close()

    def export_all(self,
                  strategy_name: str,
                  output_dir: str = 'exports') -> List[str]:
        """
        Export all data for a strategy

        Args:
            strategy_name: Name of strategy
            output_dir: Directory to save exports

        Returns:
            List of exported file paths
        """
        print(f"\n{'='*80}")
        print(f"EXPORTING ALL DATA: {strategy_name}")
        print(f"{'='*80}\n")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exported_files = []

        # Export returns
        returns_path = f"{output_dir}/{strategy_name}_returns_{timestamp}.csv"
        path = self.export_returns(strategy_name, returns_path)
        if path:
            exported_files.append(path)

        # Export positions
        positions_path = f"{output_dir}/{strategy_name}_positions_{timestamp}.csv"
        path = self.export_positions(strategy_name, positions_path)
        if path:
            exported_files.append(path)

        # Export trades
        trades_path = f"{output_dir}/{strategy_name}_trades_{timestamp}.csv"
        path = self.export_trades(strategy_name, trades_path)
        if path:
            exported_files.append(path)

        # Export alerts
        alerts_path = f"{output_dir}/{strategy_name}_alerts_{timestamp}.csv"
        path = self.export_alerts(strategy_name, alerts_path)
        if path:
            exported_files.append(path)

        print(f"\n{'='*80}")
        print(f"✅ EXPORT COMPLETE: {len(exported_files)} files")
        print(f"{'='*80}\n")

        for file_path in exported_files:
            print(f"  📄 {file_path}")

        return exported_files


# Add callback functions for Dash buttons
def create_export_callbacks(app):
    """
    Create Dash callbacks for export buttons

    Add this to your app.py to enable CSV export buttons
    """
    from dash import Output, Input
    from dash.exceptions import PreventUpdate
    import json

    @app.callback(
        Output('export-returns-download', 'data'),
        Input('export-returns-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def export_returns_callback(n_clicks):
        """Export returns when button clicked"""
        if n_clicks is None:
            raise PreventUpdate

        exporter = CSVExporter()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f'exports/returns_{timestamp}.csv'

        exporter.export_returns('equity_momentum_90d', path)

        # Return download
        return dict(content=open(path, 'r').read(), filename=f'returns_{timestamp}.csv')

    @app.callback(
        Output('export-positions-download', 'data'),
        Input('export-positions-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def export_positions_callback(n_clicks):
        """Export positions when button clicked"""
        if n_clicks is None:
            raise PreventUpdate

        exporter = CSVExporter()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f'exports/positions_{timestamp}.csv'

        exporter.export_positions('equity_momentum_90d', path)

        return dict(content=open(path, 'r').read(), filename=f'positions_{timestamp}.csv')

    @app.callback(
        Output('export-trades-download', 'data'),
        Input('export-trades-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def export_trades_callback(n_clicks):
        """Export trades when button clicked"""
        if n_clicks is None:
            raise PreventUpdate

        exporter = CSVExporter()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f'exports/trades_{timestamp}.csv'

        exporter.export_trades('equity_momentum_90d', path)

        return dict(content=open(path, 'r').read(), filename=f'trades_{timestamp}.csv')

    @app.callback(
        Output('export-alerts-download', 'data'),
        Input('export-alerts-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def export_alerts_callback(n_clicks):
        """Export alerts when button clicked"""
        if n_clicks is None:
            raise PreventUpdate

        exporter = CSVExporter()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f'exports/alerts_{timestamp}.csv'

        exporter.export_alerts('equity_momentum_90d', path)

        return dict(content=open(path, 'r').read(), filename=f'alerts_{timestamp}.csv')


if __name__ == '__main__':
    """Test CSV export functionality"""

    import sys

    exporter = CSVExporter()

    if len(sys.argv) > 1:
        strategy_name = sys.argv[1]
    else:
        strategy_name = 'equity_momentum_90d'

    print(f"\nTesting CSV export for: {strategy_name}\n")

    # Export all data
    files = exporter.export_all(strategy_name, output_dir='exports')

    print(f"\n✅ Test complete! {len(files)} files exported")
