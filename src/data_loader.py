"""
This file helps load the credit dataset into a DataFrame and perform basic validation.
"""

import pandas as pd
from src.config import DATA_PATH, TARGET_COL
from src.logger import get_logger

logger = get_logger(__name__)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
   
    logger.info("Loading data from: %s", path)

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error("Dataset not found at path: %s", path)
        raise

    logger.info("Loaded %d rows × %d columns.", *df.shape)

    # Validate target column
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    logger.info(
        "Class distribution:\n%s",
        df[TARGET_COL].value_counts().to_string(),
    )
    return df
