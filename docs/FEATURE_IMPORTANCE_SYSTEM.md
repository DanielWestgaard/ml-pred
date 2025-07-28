# Feature Importance Analysis System

## Overview
A comprehensive, modular feature importance analysis system integrated into the ml-pred pipeline. Uses multiple statistical and machine learning methods to identify the most predictive features for any hypothesis or dataset.

## Key Features

### 🎯 **Multi-Method Analysis**
- **Correlation Analysis**: Spearman correlation with target (robust to outliers)
- **Mutual Information**: Captures non-linear relationships
- **Random Forest Importance**: Tree-based feature importance
- **Lasso Regularization**: L1 penalty feature selection (regression only)
- **Permutation Importance**: Unbiased importance via feature shuffling

### 🔧 **Modular Design**
- **Standalone Script**: Can analyze any processed CSV file independently
- **Pipeline Integration**: Seamlessly integrated into EngineeringPipeline
- **Command Line Interface**: Easy access via main.py flags
- **Auto-Detection**: Automatically detects target columns and task type

### 📊 **Comprehensive Output**
- **Visual Analysis**: Multi-method comparison plots with consensus ranking
- **Detailed Reports**: Text summary with method-specific rankings
- **Consensus Scoring**: Normalized ranking across all methods
- **Automated Saving**: Results saved with timestamps to organized directories

## Usage Examples

### 1. Standalone Analysis
```bash
# Analyze specific processed dataset
python main.py --feature-importance "storage/data/capital_com/GBPUSD/processed/1m/data.csv" --fi-target "target_spread_compression_binary"

# Auto-detect target column
python main.py -fi "storage/data/capital_com/GBPUSD/processed/1m/data.csv"
```

### 2. Integrated Pipeline Analysis
```bash
# Run hypothesis pipeline with feature importance analysis
python main.py -hyp "hypothesis/spread_compression_patterns/config_process_data.yml" --fi-in-pipeline

# Regular data engineering with feature importance
python main.py -eg "path/to/raw/data.csv" --fi-in-pipeline
```

### 3. Python API Usage
```python
from data.features.feature_importance import FeatureImportanceAnalyzer

# Initialize analyzer
analyzer = FeatureImportanceAnalyzer(dataset, target_column="target_spread_compression_binary")

# Run all analysis methods
results = analyzer.analyze_all_methods()

# Get consensus ranking
consensus = analyzer.get_consensus_ranking(top_n=15)

# Generate comprehensive analysis with automatic saving
complete_results = analyzer.analyze_and_save()
```

## Optimal Pipeline Placement

The feature importance analysis is strategically placed **after feature generation but before transformation** for these reasons:

### ✅ **Advantages of This Placement**
1. **Complete Feature Set**: All features are generated and available for analysis
2. **Original Scale**: Features are in their natural scale, making importance scores more interpretable
3. **Early Insights**: Insights available before expensive transformation/selection steps
4. **Hypothesis Validation**: Can validate hypothesis-specific features immediately after generation
5. **Pipeline Efficiency**: Can inform feature selection and reduce transformation overhead

### 📍 **Integration Points**

#### EngineeringPipeline Integration
- **Location**: After `generate_selected()` or `generate_all()`, before `FeatureTransformer`
- **Trigger**: `analyze_feature_importance=True` parameter
- **Auto-Detection**: Automatically finds target columns for hypothesis-based analysis

#### Main.py Integration
- **Standalone**: `--feature-importance` / `-fi` for direct file analysis
- **Pipeline**: `--fi-in-pipeline` to enable analysis within data pipeline
- **Target**: `--fi-target` to specify target column (otherwise auto-detected)

## Output Structure

### Directory Organization
```
storage/data_engineering_stats/feature_importance/
├── feature_importance_YYYYMMDD_HHMMSS.png          # Visual comparison plot
├── importance_results_YYYYMMDD_HHMMSS.csv          # Detailed method results
├── importance_report_YYYYMMDD_HHMMSS.txt           # Text summary report  
└── consensus_ranking_YYYYMMDD_HHMMSS.csv           # Final consensus ranking
```

### Example Output
```
============================================================
TOP 10 CONSENSUS FEATURES:
============================================================
 1. compression_ratio_micro        0.9142
 2. session_atr_mean              0.6347
 3. hour_of_week_sin              0.4995
 4. atr                           0.4942
 5. volatility_ratio              0.4884
 6. volatility_zscore             0.4497
 7. hour_of_week_cos              0.4345
 8. volatility_long               0.4093
 9. minutes_to_next_session       0.3629
10. session_progress_cos          0.3277
============================================================
```

## Technical Implementation

### Class Architecture
```python
class FeatureImportanceAnalyzer:
    def __init__(self, dataset, target_column=None, task_type='auto')
    
    # Core analysis methods
    def correlation_importance() -> pd.Series
    def mutual_information_importance() -> pd.Series  
    def random_forest_importance() -> pd.Series
    def lasso_importance() -> pd.Series
    def permutation_importance_analysis() -> pd.Series
    
    # Synthesis methods
    def analyze_all_methods() -> Dict[str, pd.Series]
    def get_consensus_ranking(top_n=None) -> pd.Series
    
    # Output methods
    def plot_importance_comparison(top_n=20, save_path=None)
    def create_summary_report(top_n=15) -> str
    def analyze_and_save(output_dir) -> Dict
```

### Feature Preprocessing
- **Automatic Exclusion**: Removes target columns, timestamps, and constant features
- **Missing Value Handling**: Median imputation for numeric features
- **Feature Filtering**: Includes spread analysis features when available
- **Task Detection**: Automatically determines regression vs classification

### Robust Error Handling
- **Method Failures**: Individual methods can fail without breaking entire analysis
- **Data Issues**: Handles missing data, constant features, small datasets
- **Integration Safety**: Pipeline continues even if feature importance fails

## Integration with Existing Systems

### ModelAwareFeatureGenerator Compatibility
- **Feature Detection**: Works with all 200+ features from the enhanced generation system
- **Dependency Aware**: Analyzes features after proper dependency resolution
- **Spread-Focused**: Special handling for bid-ask spread compression features

### Hypothesis System Integration
- **Auto-Target Detection**: Finds hypothesis-specific target columns automatically
- **Config Integration**: Can be enabled via YAML configuration
- **Result Contextualization**: Results are contextualized for specific hypotheses

### Logging and Monitoring
- **Comprehensive Logging**: Detailed logs at each analysis step
- **Progress Tracking**: Clear progress indicators for long-running analyses
- **Error Reporting**: Graceful error handling with informative messages

## Best Practices

### When to Use Feature Importance Analysis
1. **After Feature Engineering**: Validate that new features are actually predictive
2. **Before Model Training**: Identify most important features to focus on
3. **Hypothesis Validation**: Confirm hypothesis-specific features are relevant
4. **Feature Selection**: Inform which features to keep/remove
5. **Model Debugging**: Understand why model performance is poor

### Performance Considerations
- **Dataset Size**: Analysis time scales with dataset size and number of features
- **Memory Usage**: Large datasets may require sampling for permutation importance
- **Parallel Processing**: Random Forest and permutation methods use parallel processing
- **Caching**: Results are automatically saved to avoid re-computation

### Interpretation Guidelines
- **Consensus Ranking**: Most reliable indicator across all methods
- **Method Agreement**: Features ranking high across multiple methods are most trustworthy
- **Domain Knowledge**: Combine statistical importance with domain expertise
- **Feature Correlation**: Consider feature redundancy when interpreting results

## Future Enhancements

### Planned Features
1. **SHAP Integration**: Add SHAP values for more interpretable feature importance
2. **Feature Interaction Analysis**: Detect important feature combinations
3. **Temporal Importance**: Track how feature importance changes over time
4. **Model-Specific Analysis**: Importance analysis for specific model types
5. **Automated Feature Selection**: Automatically filter features based on importance

### Configuration Options
1. **Method Selection**: Allow choosing specific importance methods
2. **Sampling Strategy**: Options for handling large datasets
3. **Output Customization**: Configurable output formats and destinations
4. **Threshold Setting**: Automated feature filtering based on importance thresholds

This feature importance system provides a robust, scientifically-grounded approach to understanding which features drive predictive performance in your trading models, perfectly integrated with your existing ml-pred pipeline architecture.