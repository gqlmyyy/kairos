# Trading Bot V3 - train_from_historical.py
# Complete pipeline: fetch historical data → build dataset → train XGBoost

"""
Complete training workflow from historical MT5 data:

1. Fetch historical candles from MT5 via QuantDinger (or fallback sources)
2. Process candles into training pairs with technical indicators
3. Import into database or save as JSON
4. Train XGBoost model
5. Evaluate performance

Usage:
    python train_from_historical.py --fetch-only
    python train_from_historical.py --train-only
    python train_from_historical.py  # Do everything
"""

import sys
import argparse
import os
from datetime import datetime
from typing import List, Dict, Any

from utils.logger import get_logger
from config import SYMBOLS, DB_FILE
from data.market.historical_fetcher import HistoricalDataManager
from analysis.features.historical_dataset_builder_new import (
    HistoricalDatasetBuilder,
    build_training_dataset_from_directory
)
from analysis.models.xgboost_trainer import train_model_from_db
from data.storage.database import get_conn, init_db

logger = get_logger("train_from_historical")


class HistoricalTrainingPipeline:
    """Complete pipeline for training from historical data"""
    
    def __init__(self, historical_dir: str = "data/historical"):
        self.historical_dir = historical_dir
        self.data_manager = HistoricalDataManager()
        self.dataset_builder = HistoricalDatasetBuilder()
        os.makedirs(self.historical_dir, exist_ok=True)
    
    def fetch_historical_data(
        self,
        symbols: List[str] = None,
        days_back: int = 365
    ) -> Dict[str, str]:
        """
        Step 1: Fetch historical data from MT5
        
        Returns:
            {symbol: filepath}
        """
        
        if symbols is None:
            symbols = SYMBOLS
        
        logger.info("=" * 60)
        logger.info("STEP 1: Fetching Historical Data from MT5")
        logger.info("=" * 60)
        
        files = self.data_manager.fetch_and_save_csv(
            symbols=symbols,
            output_dir=self.historical_dir
        )
        
        return files
    
    def build_training_dataset(self) -> None:
        """Step 2: Build ML-ready training dataset from CSV files"""
        
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Building Training Dataset")
        logger.info("=" * 60)
        
        build_training_dataset_from_directory(
            historical_data_dir=self.historical_dir,
            output_dir="data/training"
        )
    
    def train_model(self, min_rows: int = 50) -> None:
        """Step 3: Train XGBoost from database"""
        
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Training XGBoost Model")
        logger.info("=" * 60)
        
        try:
            model, metrics = train_model_from_db(min_rows=min_rows)
            
            if model:
                logger.info("\n✓ Model trained successfully!")
                logger.info(f"  Metrics: {metrics}")
                
                # Save model
                import json
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_file = f"models/xgb_model_{timestamp}.json"
                
                logger.info(f"  Saved to: {model_file}")
            else:
                logger.warning("⚠️  Model training did not complete")
        
        except Exception as e:
            logger.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
    
    def run_full_pipeline(
        self,
        symbols: List[str] = None,
        days_back: int = 365,
        train: bool = True
    ) -> None:
        """Execute full pipeline"""
        
        logger.info("\n" + "=" * 80)
        logger.info("HISTORICAL TRAINING PIPELINE")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        try:
            # Initialize database
            init_db()
            
            # Step 1: Fetch
            files = self.fetch_historical_data(symbols, days_back)
            
            if not files:
                logger.error("No data fetched. Exiting.")
                return
            
            # Step 2: Build dataset
            self.build_training_dataset()
            
            # Step 3: Train (optional)
            if train:
                self.train_model()
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()


def get_database_stats() -> None:
    """Show database statistics"""
    
    try:
        conn = get_conn()
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*) FROM execution_dataset")
        total = c.fetchone()[0]
        
        c.execute("""
            SELECT symbol, COUNT(*) as cnt FROM execution_dataset
            GROUP BY symbol ORDER BY cnt DESC
        """)
        by_symbol = c.fetchall()
        
        conn.close()
        
        logger.info("\nDatabase Statistics:")
        logger.info(f"  Total records: {total}")
        logger.info("  By symbol:")
        for row in by_symbol:
            logger.info(f"    {row[0]}: {row[1]}")
        
    except Exception as e:
        logger.error(f"Error reading database: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost from historical MT5 data"
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Only fetch data, don't train"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train, don't fetch (requires existing data)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of historical data to fetch (default: 365)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=SYMBOLS,
        help="Symbols to process"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics"
    )
    
    args = parser.parse_args()
    
    pipeline = HistoricalTrainingPipeline()
    
    if args.stats:
        get_database_stats()
        return
    
    if args.train_only:
        # Only build dataset and train
        pipeline.build_training_dataset()
        pipeline.train_model()
    elif args.fetch_only:
        # Only fetch data
        pipeline.fetch_historical_data(args.symbols, args.days)
    else:
        # Full pipeline
        pipeline.run_full_pipeline(
            symbols=args.symbols,
            days_back=args.days,
            train=True
        )


if __name__ == "__main__":
    main()
