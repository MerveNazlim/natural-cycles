import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_data(path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(path)

def drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the redundant 'Unnamed: 0' column if present.
    """
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        logging.info("'Unnamed: 0' column dropped.")
    return df

def report_missing_values(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """
    Report the number and percentage of missing values for each column.

    df : pd.DataFrame
    cols : list of column names, optional

    -> pd.DataFrame
    """
    # Default to all columns if none provided
    if cols is None:
        cols = df.columns

    # Count missing
    missing = df[cols].isna().sum()

    # Percentage
    missing_pct = (missing / len(df) * 100).round(2)

    # Build report
    report = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct
    })

    # Log it
    logging.info("Missing values per column:\n%s", report.to_dict('index'))

    return report


def report_extreme_cycle_values(df: pd.DataFrame, length_thresh: int = 100, std_thresh: int = 100) -> None:
    """
    Identify rows with unusually long cycle lengths or high cycle-length standard deviations.
    """
    long_cycles = df[df['average_cycle_length'] > length_thresh]
    high_std = df[df['cycle_length_std'] > std_thresh]
    logging.info("Cycles > %d days: %d rows.", length_thresh, len(long_cycles))
    logging.info("Cycle std > %d days: %d rows.", std_thresh, len(high_std))

def check_inconsistent_labels(df: pd.DataFrame, cols=None) -> dict:
    """
    Check object-type columns for inconsistent categorical labels.

    df : pd.DataFrame
    cols : list of column names to check, optional

    -> dict: column name -> list of unique values
    """
    if cols is None:
        cols = df.select_dtypes(include='object').columns

    inconsistencies = {}

    for col in cols:
        if df[col].dtype == 'object':
            unique_vals = sorted(df[col].dropna().unique())
            if len(unique_vals) > 1:
                inconsistencies[col] = unique_vals

    return inconsistencies


def convert_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert textual boolean-like columns to actual booleans.
    """
    # regular_cycle
    df['regular_cycle'] = df['regular_cycle'].map({'True': True, 'False': False})

    # Generic yes/no columns
    yes_no_cols = [
        col for col in df.columns 
        if df[col].dropna().isin(['yes', 'no']).all()
    ]
    for col in yes_no_cols:
        df[col] = df[col].map({'yes': True, 'no': False})
        logging.info("Converted column '%s' to boolean.", col)

    return df

def plot_distributions(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1]})
        
        # Histogram
        sns.histplot(df[col], kde=True, bins=30, ax=axes[0])
        axes[0].set_title(f'Distribution of {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Count')

        # Boxplot
        sns.boxplot(y=df[col], ax=axes[1])
        axes[1].set_title(f'Boxplot of {col}')
        axes[1].set_ylabel('')

        plt.tight_layout()
        plt.savefig(f'../results/distribution_{col}.png')
        plt.close()


def plot_barplots(df: pd.DataFrame, categorical_cols: Optional[List[str]] = None):
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_cols:
        plt.figure(figsize=(10, 4))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'../results/barplot_{col}.png')
        plt.close()