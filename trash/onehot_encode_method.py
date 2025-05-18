import logging

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def encode_categorical_vars(self):
    """Converting categorical/textual data into numerical format. Like One-hot encoding, label encoding."""
    try:
        categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        logging.debug(f"Categorical columns about to be one-hot encoded: {categorical_columns}")
        encoder = OneHotEncoder(sparse_output=False)

        one_hot_encoded = encoder.fit_transform(self.df[categorical_columns])
        one_hot_df = pd.DataFrame(
            one_hot_encoded, 
            columns=encoder.get_feature_names_out(categorical_columns),
            index=self.df.index  # Both having date as index
        )
        print(f"Before : {categorical_columns}")
        print(f"After : {one_hot_df.columns}")
        df_encoded = pd.concat([self.df, one_hot_df], axis=1)

        self.df = df_encoded.drop(categorical_columns, axis=1)
        logging.debug(f"Encoded data successfully!")
    except Exception as e:
        logging.error("Unable to one-hot encode dataset!")