#!/usr/bin/env python3
"""
Fix class imbalance by retraining model with clean V2 directional targets
- Removes all future data leakage from target generation  
- Uses proper three-class directional system
- Applies balanced class weights for imbalance handling
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.features.target_hypothesis import TargetHypothesis
from model.model_training import ModelTraining
from utils.logging_utils import get_logger

logger = get_logger()

def retrain_clean_model():
    """Retrain model with cleaned target generation (no data leakage)"""
    
    logger.info("🚀 Starting clean model retraining to fix class imbalance")
    logger.info("="*70)
    
    # Step 1: Load and process data
    logger.info("📊 Step 1: Loading processed data...")
    data_path = "storage/data/capital_com/GBPUSD/processed/1m/2024-10-01_2025-07-25.csv"
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return False
        
    df = pd.read_csv(data_path)
    logger.info(f"✅ Loaded {len(df):,} rows of data")
    
    # Step 2: Generate clean targets (no future data leakage)
    logger.info("🎯 Step 2: Generating clean V2 directional targets...")
    target_generator = TargetHypothesis(df, "spread_compression_patterns_v2")
    df_with_targets = target_generator.run()
    
    if df_with_targets is None:
        logger.error("❌ Target generation failed")
        return False
        
    # Verify target columns exist
    required_targets = ['target_spread_compression_v2_directional']
    missing_targets = [t for t in required_targets if t not in df_with_targets.columns]
    
    if missing_targets:
        logger.error(f"❌ Missing target columns: {missing_targets}")
        return False
        
    logger.info("✅ Clean targets generated successfully")
    
    # Step 3: Check class distribution
    target_col = 'target_spread_compression_v2_directional'
    class_dist = df_with_targets[target_col].value_counts().sort_index()
    total_samples = len(df_with_targets)
    
    logger.info("📈 Class distribution analysis:")
    for class_val, count in class_dist.items():
        percentage = count / total_samples * 100
        class_name = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(class_val, f"Class {class_val}")
        logger.info(f"  {class_name} (Class {class_val}): {count:,} samples ({percentage:.2f}%)")
    
    # Check if we have reasonable balance
    min_class_pct = (class_dist.min() / total_samples) * 100
    if min_class_pct < 1.0:
        logger.warning(f"⚠️  Smallest class is only {min_class_pct:.2f}% - severe imbalance detected")
    else:
        logger.info(f"✅ Smallest class is {min_class_pct:.2f}% - reasonable for balanced training")
    
    # Step 4: Train new model with balanced class weights
    logger.info("🤖 Step 3: Training Random Forest with balanced class weights...")
    
    # Ensure target column is integer type for classification
    target_col = 'target_spread_compression_v2_directional'
    df_with_targets[target_col] = df_with_targets[target_col].astype(int)
    
    # Save processed data temporarily  
    temp_data_path = "storage/temp_clean_training_data.csv"
    df_with_targets.to_csv(temp_data_path, index=False)
    logger.info(f"💾 Saved clean training data to {temp_data_path}")
    
    # Initialize model trainer with specific target column
    model_trainer = ModelTraining(
        model_type="random_forest",
        dataset=temp_data_path,
        target_col=target_col
    )
    
    # Train the model
    logger.info("🏋️  Starting model training...")
    success = model_trainer.run()
    
    if not success:
        logger.error("❌ Model training failed")
        return False
        
    logger.info("✅ Model training completed successfully")
    
    # Step 5: Get model info and update config
    model_path = model_trainer.model_directory
    logger.info(f"📁 New model saved to: {model_path}")
    
    # Update backtesting config
    config_path = "hypothesis/spread_compression_patterns/config_backtest.yml"
    if os.path.exists(config_path):
        # Read current config
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Replace model folder path
        old_path_line = None
        for line in config_content.split('\n'):
            if 'model_folder:' in line:
                old_path_line = line
                break
                
        if old_path_line:
            new_path_line = f"    model_folder: {model_path}"
            config_content = config_content.replace(old_path_line, new_path_line)
            
            # Write updated config
            with open(config_path, 'w') as f:
                f.write(config_content)
                
            logger.info(f"✅ Updated config to use new model: {model_path}")
        else:
            logger.warning("⚠️  Could not update config file - please update model_folder manually")
    
    # Cleanup
    if os.path.exists(temp_data_path):
        os.remove(temp_data_path)
        logger.info("🧹 Cleaned up temporary files")
    
    logger.info("🎉 Clean model retraining completed successfully!")
    logger.info("="*70)
    logger.info("Next steps:")
    logger.info("1. Run backtesting with updated config")  
    logger.info("2. Monitor for improved performance vs previous -5% return")
    logger.info("3. Check that model uses proper three-class predictions")
    
    return True

if __name__ == "__main__":
    success = retrain_clean_model()
    if not success:
        logger.error("❌ Clean model retraining failed")
        sys.exit(1)
    else:
        logger.info("✅ All done!")