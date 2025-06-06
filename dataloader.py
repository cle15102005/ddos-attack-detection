import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    dropping unnecessary columns, fixing categorical issues,
    handling inf/nan values, and converting the 'Label' to binary (0 for benign, 1 for attack).
    """
    
    # Drop unnecessary columns
    drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Protocol']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

    # Fix any categorical issues
    for col in df.select_dtypes(include='category').columns:
        df[col] = df[col].astype(str)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Convert label to binary
    df['Label'] = df['Label'].astype(str)
    df['Label'] = df['Label'].apply(lambda x: 0 if x.lower() == 'benign' else 1)

    return df

def load_train_data(parquet_path: str = 'smallCiCDDoS/NTP-testing.parquet', test_size: float = 0.5, random_state: int = 42):
    """
    Load the train dataset.
    S1 contains data for fitting the GEM detector, and S2 for scoring baseline.
    Both S1 and S2 consist only of benign samples.

    Args:
        parquet_path (str): Path to the parquet file.
        test_size (float): The proportion of the dataset to include in the S2 split.
        random_state (int): Seed for random number generation for reproducibility.
    """
    
    df = pd.read_parquet(parquet_path)
    print("✅ Raw data loaded for training:", df.shape)

    df = _preprocess_data(df.copy()) # Use a copy to avoid modifying original df for other loaders
    print("✅ Cleaned data for training:", df.shape)
    print("Label distribution for training data:\n", df['Label'].value_counts())

    # Extract only benign samples for GEM offline phase
    benign_df = df[df['Label'] == 0].drop(columns=['Label'])

    # Partition benign data into S1 (training) and S2 (scoring baseline)
    S1_df, S2_df = train_test_split(benign_df, test_size=test_size, random_state=random_state)
    print(f"✅ Partitioned benign data: S1={S1_df.shape}, S2={S2_df.shape}")

    # Initialize and fit StandardScaler on S1, then transform S1 and S2
    scaler = StandardScaler()
    S1_scaled = scaler.fit_transform(S1_df)
    S2_scaled = scaler.transform(S2_df)
    print("✅ Data scaled using StandardScaler fitted on S1.")

    return S1_scaled, S2_scaled, scaler

def load_test_data(parquet_path: str = 'smallCiCDDoS/NTP-testing.parquet'):
    """
    Load the test dataset.

    Args:
        parquet_path (str): Path to the parquet file.
    """
    df = pd.read_parquet(parquet_path)
    print("✅ Raw data loaded for testing:", df.shape)

    df = _preprocess_data(df.copy()) # Use a copy to avoid modifying original df for other loaders
    print("✅ Cleaned data for testing:", df.shape)
    print("Label distribution for testing data:\n", df['Label'].value_counts())

    X_all = df.drop(columns=['Label'])
    y_all = df['Label']

    return X_all.to_numpy(), y_all.to_numpy()
