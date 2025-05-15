import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder

def preprocess_features_for_xgboost(df, target_col=None, enable_categorical=False):
    """
    Preprocess features for XGBoost, handling all object and category types dynamically.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe with features
    target_col : str, optional
        Name of target column to exclude from processing
    enable_categorical : bool, default=False
        Whether to use XGBoost's native categorical support (requires XGBoost 1.5+)
        
    Returns:
    --------
    pandas DataFrame
        Processed dataframe ready for XGBoost
    """
    # First, check for and fix duplicate columns
    df_processed = df.copy()
    
    # Check for duplicate columns and fix them
    # duplicate_cols = df_processed.columns[df_processed.columns.duplicated()].tolist()
    # if duplicate_cols:
    #     print(f"Warning: Found duplicate columns: {duplicate_cols}")
    #     # Rename duplicates with a suffix
    #     for col in duplicate_cols:
    #         # Get all occurrences of the duplicated column
    #         cols = df_processed.columns.get_indexer_for([col])
    #         # Rename all but the first occurrence
    #         for i, idx in enumerate(cols[1:], 1):
    #             new_name = f"{col}_{i}"
    #             df_processed.columns.values[idx] = new_name
    #             print(f"Renamed duplicate column '{col}' to '{new_name}'")
    
    # 1. Handle fft_dominant_periods (list stored as string)
    if 'fft_dominant_periods' in df_processed.columns:
        try:
            # Convert string representation of list to actual values
            df_processed['fft_dominant_periods'] = df_processed['fft_dominant_periods'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            
            # Extract specific components (first three elements if available)
            max_components = 3
            for i in range(max_components):
                df_processed[f'fft_dom_period_{i+1}'] = df_processed['fft_dominant_periods'].apply(
                    lambda x: float(x[i]) if isinstance(x, (list, tuple)) and len(x) > i else np.nan
                )
            
            # Calculate mean and max of components as additional features
            df_processed['fft_dom_period_mean'] = df_processed['fft_dominant_periods'].apply(
                lambda x: np.mean(x) if isinstance(x, (list, tuple)) and len(x) > 0 else np.nan
            )
            df_processed['fft_dom_period_max'] = df_processed['fft_dominant_periods'].apply(
                lambda x: np.max(x) if isinstance(x, (list, tuple)) and len(x) > 0 else np.nan
            )
            
            # Drop the original column
            df_processed = df_processed.drop('fft_dominant_periods', axis=1)
        
        except Exception as e:
            print(f"Error processing fft_dominant_periods: {e}")
            # Fallback: Label encode as a last resort
            le = LabelEncoder()
            df_processed['fft_dom_encoded'] = le.fit_transform(df_processed['fft_dominant_periods'])
            df_processed = df_processed.drop('fft_dominant_periods', axis=1)
    
    # 2. Process other categorical features
    cat_columns = []
    for col in df_processed.columns:
        if col == target_col:
            continue
        # Check if it's a single column (not duplicated)
        if isinstance(df_processed[col], pd.Series):
            if df_processed[col].dtype == 'object' or df_processed[col].dtype.name == 'category':
                cat_columns.append(col)
    
    if enable_categorical:
        # Option 1: Use XGBoost's native categorical support
        for col in cat_columns:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].astype('category')
    else:
        # Option 2: Encode categorical features
        for col in cat_columns:
            n_unique = df_processed[col].nunique()
            
            if n_unique <= 10:  # Low cardinality -> one-hot encoding
                # Create dummies with better column names for clarity
                dummies = pd.get_dummies(
                    df_processed[col], 
                    prefix=col, 
                    prefix_sep='_',
                    drop_first=True  # Remove one category to avoid multicollinearity
                )
                df_processed = pd.concat([df_processed, dummies], axis=1)
            else:  # High cardinality -> label encoding
                le = LabelEncoder()
                df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            
            # Drop original categorical column
            df_processed = df_processed.drop(col, axis=1)
    
    # 3. Final check for any remaining non-numeric columns
    non_numeric_cols = []
    for col in df_processed.columns:
        if col == target_col:
            continue
        # Check if it's a single column (not duplicated)
        try:
            is_numeric = pd.api.types.is_numeric_dtype(df_processed[col])
            if not is_numeric:
                non_numeric_cols.append(col)
        except Exception as e:
            print(f"Error checking column {col}: {e}")
            non_numeric_cols.append(col)
    
    # Drop non-numeric columns
    if non_numeric_cols:
        print(f"Warning: Dropping non-numeric columns: {non_numeric_cols}")
        df_processed = df_processed.drop(non_numeric_cols, axis=1)
    
    return df_processed