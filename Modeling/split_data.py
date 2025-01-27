import pandas as pd
from typing import Dict, Tuple, List
import numpy as np

def filter_single_category_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows with multiple categories"""
    return df[
        df['Category'].notna() & 
        df['Category'].str.split(',').apply(lambda x: len(x) == 1 if isinstance(x, list) else False)
    ]

def get_label_splits(label_df: pd.DataFrame, 
                    train_size: int, 
                    test_size: int, 
                    val_size: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data for a single label into train/test/val sets"""
    # Shuffle the data
    label_df = label_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Adjust sizes based on available data
    total_size = len(label_df)
    train_size = min(train_size, total_size)
    test_size = min(test_size, total_size - train_size)
    val_size = min(val_size, total_size - train_size - test_size)
    
    # Split the data
    train = label_df.iloc[:train_size]
    test = label_df.iloc[train_size:train_size+test_size]
    val = label_df.iloc[train_size+test_size:train_size+test_size+val_size]
    
    return train, test, val

def split_dataset(df: pd.DataFrame,
                 train_label_counts: Dict[str, int] = None,
                 test_label_counts: Dict[str, int] = None,
                 val_label_counts: Dict[str, int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, test, and validation sets based on specified label counts
    """
    # Default label counts if not provided
    if train_label_counts is None:
        train_label_counts = {
            'IMMORAL_NONE': 4000, 'CENSURE': 4000, 'SEXUAL': 4000,
            'DISCRIMINATION': 4000, 'HATE': 4000, 'VIOLENCE': 3000,
            'ABUSE': 3000, 'CRIME': 700
        }
    
    if test_label_counts is None:
        test_label_counts = {
            'IMMORAL_NONE': 2000, 'CENSURE': 2000, 'SEXUAL': 2000,
            'DISCRIMINATION': 2000, 'HATE': 2000, 'VIOLENCE': 2000,
            'ABUSE': 1000, 'CRIME': 300
        }
    
    if val_label_counts is None:
        val_label_counts = {
            'IMMORAL_NONE': 2000, 'CENSURE': 2000, 'SEXUAL': 2000,
            'DISCRIMINATION': 2000, 'HATE': 2000, 'VIOLENCE': 1000,
            'ABUSE': 900, 'CRIME': 300
        }
    
    # Filter data
    df_filtered = filter_single_category_data(df)
    
    # Prepare lists to store split data
    train_data, test_data, val_data = [], [], []
    
    # Process each unique label
    for label in df_filtered['Category'].unique():
        label_df = df_filtered[df_filtered['Category'] == label]
        
        train_sample, test_sample, val_sample = get_label_splits(
            label_df,
            train_label_counts.get(label, 0),
            test_label_counts.get(label, 0),
            val_label_counts.get(label, 0)
        )
        
        train_data.append(train_sample)
        test_data.append(test_sample)
        val_data.append(val_sample)
    
    # Combine and shuffle datasets
    train_df = pd.concat(train_data).sample(frac=1, random_state=42)
    test_df = pd.concat(test_data).sample(frac=1, random_state=42)
    val_df = pd.concat(val_data).sample(frac=1, random_state=42)
    
    # Print statistics
    print("\nDataset split statistics:")
    print(f"Train dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")
    print(f"Validation dataset shape: {val_df.shape}")
    
    print("\nTrain label distribution:")
    print(train_df['Category'].value_counts())
    print("\nTest label distribution:")
    print(test_df['Category'].value_counts())
    print("\nValidation label distribution:")
    print(val_df['Category'].value_counts())
    
    return train_df, test_df, val_df

if __name__ == "__main__":
    # Example usage for standalone execution
    input_file = "./data/combined_data.csv"
    df = pd.read_csv(input_file)
    
    train_df, test_df, val_df = split_dataset(df)
    
    # Save results
    # train_df.to_csv('train_split.csv', encoding='utf-8-sig', index=False)
    # test_df.to_csv('test_split.csv', encoding='utf-8-sig', index=False)
    # val_df.to_csv('val_split.csv', encoding='utf-8-sig', index=False)