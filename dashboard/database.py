"""
Database Models for Trading Dashboard

Stores historical performance data, positions, trades, alerts, and regimes
for comprehensive strategy monitoring and analysis.
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()


class PerformanceMetric(Base):
    """Daily performance metrics for each strategy"""
    __tablename__ = 'performance_metrics'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, index=True)
    strategy_name = Column(String(100), nullable=False, index=True)

    # Returns
    daily_return = Column(Float)
    cumulative_return = Column(Float)
    annual_return = Column(Float)

    # Risk Metrics
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    calmar_ratio = Column(Float)
    volatility = Column(Float)

    # Drawdown
    drawdown = Column(Float)
    max_drawdown = Column(Float)

    # Position Info
    num_positions = Column(Integer)
    total_exposure = Column(Float)
    net_exposure = Column(Float)

    # Attribution Components (from our analysis)
    benchmark_return = Column(Float)
    selection_alpha = Column(Float)
    regime_contribution = Column(Float)
    transaction_costs = Column(Float)

    # Timestamp
    updated_at = Column(DateTime, default=datetime.utcnow)


class Position(Base):
    """Current and historical positions"""
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, index=True)
    strategy_name = Column(String(100), nullable=False, index=True)

    symbol = Column(String(20), nullable=False)
    weight = Column(Float)
    quantity = Column(Float)
    price = Column(Float)
    market_value = Column(Float)

    # P&L
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float)

    # Risk
    position_volatility = Column(Float)
    position_beta = Column(Float)

    updated_at = Column(DateTime, default=datetime.utcnow)


class Trade(Base):
    """Trade execution log"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    strategy_name = Column(String(100), nullable=False, index=True)

    symbol = Column(String(20), nullable=False)
    side = Column(String(10))  # 'BUY' or 'SELL'
    quantity = Column(Float)
    price = Column(Float)

    # Costs
    commission = Column(Float)
    slippage = Column(Float)
    total_cost = Column(Float)

    # Context
    regime_at_trade = Column(String(20))
    reason = Column(Text)  # Why this trade was made

    updated_at = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    """System alerts and warnings"""
    __tablename__ = 'alerts'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    strategy_name = Column(String(100), index=True)

    severity = Column(String(20))  # 'INFO', 'WARNING', 'CRITICAL'
    category = Column(String(50))  # 'DRAWDOWN', 'SHARPE', 'REGIME', 'SYSTEM'
    message = Column(Text)

    # Alert details
    threshold_value = Column(Float)
    actual_value = Column(Float)

    # Status
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow)


class Regime(Base):
    """Market regime detection history"""
    __tablename__ = 'regimes'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, index=True)
    strategy_name = Column(String(100), nullable=False, index=True)

    regime = Column(String(20))  # 'BULL', 'BEAR', 'SIDEWAYS', 'CORRECTION', 'CRISIS'
    confidence = Column(Float)

    # Indicators used
    trend_indicator = Column(Float)
    volatility_indicator = Column(Float)
    drawdown_indicator = Column(Float)

    # Trading recommendation
    recommended_exposure = Column(Float)

    updated_at = Column(DateTime, default=datetime.utcnow)


class StrategyConfig(Base):
    """Strategy configuration snapshots"""
    __tablename__ = 'strategy_configs'

    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(100), nullable=False, index=True)
    version = Column(String(20))

    # Configuration as JSON-like text
    config_json = Column(Text)

    # Performance targets
    target_sharpe = Column(Float)
    target_return = Column(Float)
    max_drawdown_limit = Column(Float)

    # Status
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class WalkForwardResult(Base):
    """Walk-forward validation results over time"""
    __tablename__ = 'walk_forward_results'

    id = Column(Integer, primary_key=True)
    run_date = Column(DateTime, nullable=False, index=True)
    strategy_name = Column(String(100), nullable=False, index=True)

    # Train period
    train_start = Column(DateTime)
    train_end = Column(DateTime)
    train_sharpe = Column(Float)
    train_return = Column(Float)

    # Test period
    test_start = Column(DateTime)
    test_end = Column(DateTime)
    test_sharpe = Column(Float)
    test_return = Column(Float)

    # Degradation
    sharpe_degradation = Column(Float)
    return_degradation = Column(Float)

    # Status
    passed = Column(Boolean)

    created_at = Column(DateTime, default=datetime.utcnow)


# Database initialization
class DatabaseManager:
    """Manages database connection and operations"""

    def __init__(self, db_path='data/dashboard.db'):
        """Initialize database manager"""
        self.db_path = db_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else 'data', exist_ok=True)

        # Create engine
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create session maker
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self):
        """Get a new database session"""
        return self.Session()

    def add_performance_metric(self, session, **kwargs):
        """Add a performance metric record"""
        metric = PerformanceMetric(**kwargs)
        session.add(metric)
        session.commit()
        return metric

    def add_position(self, session, **kwargs):
        """Add a position record"""
        position = Position(**kwargs)
        session.add(position)
        session.commit()
        return position

    def add_trade(self, session, **kwargs):
        """Add a trade record"""
        trade = Trade(**kwargs)
        session.add(trade)
        session.commit()
        return trade

    def add_alert(self, session, severity, category, message, **kwargs):
        """Add an alert"""
        alert = Alert(
            severity=severity,
            category=category,
            message=message,
            **kwargs
        )
        session.add(alert)
        session.commit()
        return alert

    def get_latest_metrics(self, session, strategy_name, days=30):
        """Get latest performance metrics"""
        return session.query(PerformanceMetric).filter(
            PerformanceMetric.strategy_name == strategy_name
        ).order_by(PerformanceMetric.date.desc()).limit(days).all()

    def get_current_positions(self, session, strategy_name):
        """Get current positions for a strategy"""
        latest_date = session.query(Position.date).filter(
            Position.strategy_name == strategy_name
        ).order_by(Position.date.desc()).first()

        if latest_date:
            return session.query(Position).filter(
                Position.strategy_name == strategy_name,
                Position.date == latest_date[0]
            ).all()
        return []

    def get_active_alerts(self, session, strategy_name=None):
        """Get active (unresolved) alerts"""
        query = session.query(Alert).filter(Alert.resolved == False)

        if strategy_name:
            query = query.filter(Alert.strategy_name == strategy_name)

        return query.order_by(Alert.timestamp.desc()).all()

    def get_latest_regime(self, session, strategy_name):
        """Get latest regime detection"""
        return session.query(Regime).filter(
            Regime.strategy_name == strategy_name
        ).order_by(Regime.date.desc()).first()


# Initialize global database manager
db_manager = DatabaseManager()


if __name__ == '__main__':
    """Test database setup"""
    print("Initializing trading dashboard database...")

    db = DatabaseManager('data/dashboard.db')
    session = db.get_session()

    print(f"✅ Database created at: data/dashboard.db")
    print(f"✅ Tables created: {len(Base.metadata.tables)}")

    for table_name in Base.metadata.tables.keys():
        print(f"   - {table_name}")

    # Test insert
    test_metric = db.add_performance_metric(
        session,
        date=datetime.now(),
        strategy_name='test_strategy',
        daily_return=0.01,
        sharpe_ratio=1.5,
        drawdown=-0.05
    )

    print(f"\n✅ Test record inserted: ID {test_metric.id}")

    # Clean up test
    session.delete(test_metric)
    session.commit()

    print("✅ Database ready for dashboard!")

    session.close()
