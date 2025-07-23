#!/usr/bin/env python3
"""
Storage Migration Script
Reorganizes historical data from provider-based to symbol-based structure.

Current: storage/data/capital_com/{raw,processed}/filename
New:     storage/data/capital_com/{SYMBOL}/{raw,processed}/{timeframe}/filename

Author: AI Assistant
Date: 2025-07-23
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StorageMigrator:
    def __init__(self, storage_root: str = "storage"):
        self.storage_root = Path(storage_root)
        self.capital_com_path = self.storage_root / "data" / "capital_com"
        self.backup_path = self.storage_root / "data" / "capital_com_backup"
        
    def parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """
        Parse filename to extract symbol, timeframe, and date range.
        
        Examples:
        - raw_BTCUSD_m_20250501_20250718.csv -> {symbol: BTCUSD, timeframe: 1m, start: 2025-05-01, end: 2025-07-18}
        - processed_GBPUSD_m5_20240101_20250101.csv -> {symbol: GBPUSD, timeframe: 5m, start: 2024-01-01, end: 2025-01-01}
        """
        # Pattern to match various filename formats
        patterns = [
            # Pattern: (raw|processed)_SYMBOL_TIMEFRAME_STARTDATE_ENDDATE.csv
            r'^(raw|processed)_([A-Z]+)_([m0-9]+)_(\d{8})_(\d{8})\.csv$',
            # Pattern: SYMBOL_TIMEFRAME_STARTDATE_ENDDATE.csv (no prefix)
            r'^([A-Z]+)_([m0-9]+)_(\d{8})_(\d{8})\.csv$'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                if len(match.groups()) == 5:  # With raw/processed prefix
                    _, symbol, timeframe, start_date, end_date = match.groups()
                else:  # Without prefix
                    symbol, timeframe, start_date, end_date = match.groups()
                
                # Normalize timeframe
                timeframe_map = {
                    'm': '1m',
                    'm1': '1m',
                    'm5': '5m',
                    'm15': '15m',
                    'm30': '30m',
                    'h': '1h',
                    'h1': '1h',
                    'h4': '4h',
                    'd': '1d',
                    'd1': '1d'
                }
                
                normalized_timeframe = timeframe_map.get(timeframe, timeframe)
                
                # Format dates
                formatted_start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
                formatted_end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
                
                return {
                    'symbol': symbol,
                    'timeframe': normalized_timeframe,
                    'start_date': formatted_start,
                    'end_date': formatted_end,
                    'new_filename': f"{formatted_start}_{formatted_end}.csv"
                }
        
        logger.warning(f"Could not parse filename: {filename}")
        return None
    
    def create_backup(self) -> None:
        """Create backup of current structure before migration."""
        if self.backup_path.exists():
            logger.info(f"Removing existing backup at {self.backup_path}")
            shutil.rmtree(self.backup_path)
        
        logger.info(f"Creating backup at {self.backup_path}")
        shutil.copytree(self.capital_com_path, self.backup_path)
    
    def create_new_structure(self) -> None:
        """Create new directory structure."""
        # Get all unique symbols from existing files
        symbols = set()
        
        for data_type in ['raw', 'processed']:
            data_path = self.capital_com_path / data_type
            if data_path.exists():
                for file in data_path.glob("*.csv"):
                    parsed = self.parse_filename(file.name)
                    if parsed:
                        symbols.add(parsed['symbol'])
        
        logger.info(f"Found symbols: {sorted(symbols)}")
        
        # Create new structure for each symbol
        for symbol in symbols:
            for data_type in ['raw', 'processed']:
                for timeframe in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']:
                    new_path = self.capital_com_path / symbol / data_type / timeframe
                    new_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Created new directory structure")
    
    def migrate_files(self) -> Dict[str, List[str]]:
        """Migrate files to new structure."""
        migration_log = {'moved': [], 'failed': [], 'skipped': []}
        
        for data_type in ['raw', 'processed']:
            old_path = self.capital_com_path / data_type
            if not old_path.exists():
                continue
                
            for file in old_path.glob("*.csv"):
                parsed = self.parse_filename(file.name)
                if not parsed:
                    migration_log['failed'].append(f"Could not parse: {file.name}")
                    continue
                
                # Skip test files
                if 'testing' in file.name.lower():
                    migration_log['skipped'].append(f"Skipped test file: {file.name}")
                    continue
                
                # Determine new location
                new_dir = self.capital_com_path / parsed['symbol'] / data_type / parsed['timeframe']
                new_file = new_dir / parsed['new_filename']
                
                try:
                    # Move file
                    shutil.move(str(file), str(new_file))
                    migration_log['moved'].append(f"{file.name} -> {parsed['symbol']}/{data_type}/{parsed['timeframe']}/{parsed['new_filename']}")
                    logger.info(f"Moved: {file.name} -> {new_file.relative_to(self.capital_com_path)}")
                except Exception as e:
                    migration_log['failed'].append(f"Failed to move {file.name}: {str(e)}")
                    logger.error(f"Failed to move {file.name}: {str(e)}")
        
        return migration_log
    
    def cleanup_old_structure(self) -> None:
        """Remove empty old directories."""
        for data_type in ['raw', 'processed']:
            old_path = self.capital_com_path / data_type
            if old_path.exists() and not any(old_path.iterdir()):
                logger.info(f"Removing empty directory: {old_path}")
                old_path.rmdir()
    
    def update_storage_readme(self) -> None:
        """Update storage README with new structure."""
        readme_path = self.storage_root / "README.md"
        if not readme_path.exists():
            return
        
        new_structure = """# About /storage
This folder is used for storage of different purposes.

## Structure
```
├── data/                       # Storing datasets
│   ├── capital_com/            # Datasets from Capital.com
│   │   ├── BTCUSD/             # Bitcoin to USD data
│   │   │   ├── raw/            # Raw data from API
│   │   │   │   ├── 1m/         # 1-minute timeframe
│   │   │   │   ├── 5m/         # 5-minute timeframe
│   │   │   │   └── 1h/         # 1-hour timeframe
│   │   │   └── processed/      # Processed data through pipeline
│   │   │       ├── 1m/
│   │   │       ├── 5m/
│   │   │       └── 1h/
│   │   ├── GBPUSD/             # British Pound to USD data
│   │   │   ├── raw/
│   │   │   │   ├── 1m/
│   │   │   │   ├── 5m/
│   │   │   │   └── 1h/
│   │   │   └── processed/
│   │   │       ├── 1m/
│   │   │       ├── 5m/
│   │   │       └── 1h/
│   │   └── saved_responses/    # API responses for debugging
│   ├── alpha_vantage/          # Datasets from Alpha Vantage
│   │   ├── processed/
│   │   ├── raw/
│   │   └── saved_responses/
│   └── ...
│
├── data_engineering_stats/     # Statistics and plots during data engineering pipeline
│   ├── cleaned_data/           # Plots of dataset after cleaning
│   ├── fe_generation/
│   └── fe_selection/
│
├── models/                     # Trained models and metadata
│   ├── <datetimenow>/          # Name of directory for model will be of now: Eg. '2025-06-09_20:19:29/'
│   │   ├── trained_model.pkl   # Actual trained model. Eg. 'xgboost.pkl'
│   │   ├── metadata.yml        # Information (metadata) about model training, location, parameters, etc.
│   │   └── evaluation/         # Evaluation of model and plots of its predictions. Created in evaluation.py (plot method).
│   │       ├── plot.pdf        # eg.
│   │       └── evaluation.pdf  # eg.
│   └── ...
│
├── backtesting_results/        # Results from backtesting
│
└── README.md                   # Documentation related to storage/
```

## File Naming Convention
Data files follow the pattern: `YYYY-MM-DD_YYYY-MM-DD.csv` (start_date_end_date.csv)

Example: `2025-05-01_2025-07-18.csv` contains data from May 1, 2025 to July 18, 2025.

## Organization Benefits
- **Symbol-based**: Easy to find data for specific trading pairs
- **Timeframe separation**: Clear organization by data granularity
- **Scalable**: Simple to add new symbols and timeframes
- **Consistent naming**: Standardized date format for easy sorting
- **Intuitive navigation**: Mirrors how traders think about market data
"""
        
        with open(readme_path, 'w') as f:
            f.write(new_structure)
        
        logger.info("Updated storage README.md")
    
    def generate_migration_report(self, migration_log: Dict[str, List[str]]) -> None:
        """Generate detailed migration report."""
        report_path = self.storage_root / "migration_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("STORAGE MIGRATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Migration completed at: {pd.Timestamp.now()}\n\n")
            
            f.write(f"FILES SUCCESSFULLY MOVED ({len(migration_log['moved'])}):\n")
            f.write("-" * 40 + "\n")
            for item in migration_log['moved']:
                f.write(f"✓ {item}\n")
            
            f.write(f"\nFILES SKIPPED ({len(migration_log['skipped'])}):\n")
            f.write("-" * 40 + "\n")
            for item in migration_log['skipped']:
                f.write(f"- {item}\n")
            
            f.write(f"\nFILES FAILED ({len(migration_log['failed'])}):\n")
            f.write("-" * 40 + "\n")
            for item in migration_log['failed']:
                f.write(f"✗ {item}\n")
        
        logger.info(f"Migration report saved to: {report_path}")
    
    def run_migration(self) -> None:
        """Run complete migration process."""
        logger.info("Starting storage migration...")
        
        # Step 1: Create backup
        logger.info("Step 1: Creating backup...")
        self.create_backup()
        
        # Step 2: Create new structure
        logger.info("Step 2: Creating new directory structure...")
        self.create_new_structure()
        
        # Step 3: Migrate files
        logger.info("Step 3: Migrating files...")
        migration_log = self.migrate_files()
        
        # Step 4: Cleanup
        logger.info("Step 4: Cleaning up old structure...")
        self.cleanup_old_structure()
        
        # Step 5: Update documentation
        logger.info("Step 5: Updating documentation...")
        self.update_storage_readme()
        
        # Step 6: Generate report
        logger.info("Step 6: Generating migration report...")
        self.generate_migration_report(migration_log)
        
        logger.info("Migration completed successfully!")
        logger.info(f"Backup created at: {self.backup_path}")
        logger.info(f"Files moved: {len(migration_log['moved'])}")
        logger.info(f"Files failed: {len(migration_log['failed'])}")
        logger.info(f"Files skipped: {len(migration_log['skipped'])}")


if __name__ == "__main__":
    import pandas as pd
    
    migrator = StorageMigrator()
    
    # Ask for confirmation
    print("This script will reorganize your storage structure.")
    print("A backup will be created before any changes are made.")
    print(f"Backup location: {migrator.backup_path}")
    
    confirm = input("\nProceed with migration? (y/N): ").strip().lower()
    if confirm == 'y':
        migrator.run_migration()
    else:
        print("Migration cancelled.")