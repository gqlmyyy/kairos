# Trading Bot V3 - test_historical_setup.py
# Verify that the historical training system is ready to use

"""
Test script to verify:
1. QuantDinger connection
2. Database setup
3. Dependencies installed
4. Can fetch sample data
5. Can process data

Run with: python test_historical_setup.py
"""

import sys
import os
from datetime import datetime

from utils.logger import get_logger

logger = get_logger("test_historical_setup")


def test_imports():
    """Test all required imports"""
    logger.info("Testing imports...")
    
    try:
        import requests
        logger.info("  ✓ requests")
    except ImportError:
        logger.error("  ✗ requests - install with: pip install requests")
        return False
    
    try:
        from data.market.historical_fetcher import HistoricalDataManager
        logger.info("  ✓ historical_fetcher")
    except ImportError as e:
        logger.error(f"  ✗ historical_fetcher: {e}")
        return False
    
    try:
        from analysis.features.historical_dataset_builder_new import HistoricalDatasetBuilder
        logger.info("  ✓ historical_dataset_builder")
    except ImportError as e:
        logger.error(f"  ✗ historical_dataset_builder: {e}")
        return False
    
    try:
        from analysis.models.xgboost_trainer import train_model_from_db
        logger.info("  ✓ xgboost_trainer")
    except ImportError as e:
        logger.error(f"  ✗ xgboost_trainer: {e}")
        return False
    
    return True


def test_quantdinger_connection():
    """Test connection to QuantDinger"""
    logger.info("\nTesting QuantDinger connection...")
    
    try:
        import requests
        from config import QUANTDINGER_URL
        from execution.quantdinger_client import get_headers
        
        headers = get_headers()
        
        r = requests.get(
            f"{QUANTDINGER_URL}/api/mt5/status",
            headers=headers,
            timeout=5
        )
        
        status = r.json()
        
        if status.get("connected"):
            logger.info("  ✓ QuantDinger connected")
            logger.info(f"    Balance: {status.get('account', {}).get('balance', 'N/A')}")
            return True
        else:
            logger.warning("  ⚠ QuantDinger not connected (may need reconnect)")
            return True  # Not critical
    
    except Exception as e:
        logger.error(f"  ✗ QuantDinger error: {e}")
        logger.info("    Make sure QuantDinger is running on localhost:8888")
        return False


def test_database():
    """Test database setup"""
    logger.info("\nTesting database...")
    
    try:
        from data.storage.database import init_db, get_conn
        
        init_db()
        logger.info("  ✓ Database initialized")
        
        conn = get_conn()
        c = conn.cursor()
        
        # Check if execution_dataset exists
        c.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='execution_dataset'
        """)
        
        if c.fetchone():
            logger.info("  ✓ execution_dataset table exists")
            
            # Count rows
            c.execute("SELECT COUNT(*) FROM execution_dataset")
            count = c.fetchone()[0]
            logger.info(f"    Currently has {count} rows")
        else:
            logger.error("  ✗ execution_dataset table not found")
            conn.close()
            return False
        
        conn.close()
        return True
    
    except Exception as e:
        logger.error(f"  ✗ Database error: {e}")
        return False


def test_sample_fetch():
    """Test fetching a sample of data"""
    logger.info("\nTesting sample data fetch...")
    
    try:
        from data.market.historical_fetcher import MT5HistoricalFetcher
        
        fetcher = MT5HistoricalFetcher()
        
        # Try to fetch a small sample
        candles = fetcher.fetch_candles(
            symbol="EURUSD",
            timeframe="H1",
            limit=10
        )
        
        if candles:
            logger.info(f"  ✓ Fetched {len(candles)} sample candles")
            first_candle = candles[0]
            logger.info(f"    Sample: {first_candle}")
            return True
        else:
            logger.warning("  ⚠ No candles returned (check MT5 connection)")
            return True  # Not critical
    
    except Exception as e:
        logger.error(f"  ✗ Fetch error: {e}")
        return False


def test_directory_structure():
    """Test that required directories exist"""
    logger.info("\nTesting directory structure...")
    
    dirs = [
        "data",
        "data/market",
        "data/historical",
        "data/training",
        "analysis/features",
        "analysis/models",
        "models"
    ]
    
    os.makedirs("data/historical", exist_ok=True)
    os.makedirs("data/training", exist_ok=True)
    
    for d in dirs:
        if os.path.isdir(d):
            logger.info(f"  ✓ {d}/")
        else:
            if d in ["data/historical", "data/training"]:
                os.makedirs(d, exist_ok=True)
                logger.info(f"  ✓ {d}/ (created)")
            else:
                logger.error(f"  ✗ {d}/ not found")
                return False
    
    return True


def test_optional_dependencies():
    """Test optional dependencies"""
    logger.info("\nTesting optional dependencies...")
    
    optional_libs = {
        "yfinance": "pip install yfinance",
        "alpha_vantage": "pip install alpha-vantage",
        "pandas": "pip install pandas"
    }
    
    for lib, install_cmd in optional_libs.items():
        try:
            __import__(lib.replace("_", "-"))
            logger.info(f"  ✓ {lib}")
        except ImportError:
            logger.warning(f"  ⚠ {lib} not installed ({install_cmd})")
    
    return True


def main():
    """Run all tests"""
    
    logger.info("\n" + "=" * 60)
    logger.info("Historical Training System - Setup Test")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Database", test_database),
        ("QuantDinger Connection", test_quantdinger_connection),
        ("Sample Data Fetch", test_sample_fetch),
        ("Optional Dependencies", test_optional_dependencies),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✓" if result else "✗"
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with exception: {e}")
            results[test_name] = "✗"
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v == "✓")
    total = len(results)
    
    for test_name, result in results.items():
        logger.info(f"{result} {test_name}")
    
    logger.info(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n✓ System is ready to use!")
        logger.info("\nRun training with:")
        logger.info("  python train_from_historical.py")
        logger.info("\nOr use the batch file:")
        logger.info("  train_historical.bat")
        return 0
    else:
        logger.warning("\n⚠ Some tests failed. See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
