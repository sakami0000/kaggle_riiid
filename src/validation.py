from typing import Tuple

import numpy as np
import pandas as pd


def virtual_time_split(train_df: pd.DataFrame,
                       valid_size: int,
                       epoch_valid_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """validation split based on virtual timestamp.
    
    References
    ----------
    - https://www.kaggle.com/its7171/cv-strategy
    """
    max_timestamp_user = train_df.groupby('user_id')['timestamp'].max()
    virtual_timestamp_user = np.random.rand(len(max_timestamp_user)) * (max_timestamp_user.max() - max_timestamp_user)
    virtual_timestamp = virtual_timestamp_user.reindex(train_df['user_id']).reset_index(drop=True) + train_df['timestamp']

    quantile = 1 - valid_size / len(virtual_timestamp)
    valid_split_time = virtual_timestamp.quantile(quantile)
    valid_idx = np.where(virtual_timestamp > valid_split_time)[0]
    train_idx = np.delete(range(len(train_df)), valid_idx)

    # idx for epoch validation
    quantile = 1 - (valid_size - epoch_valid_size) / len(virtual_timestamp)
    epoch_valid_split_time = virtual_timestamp.quantile(quantile)
    epoch_valid_idx = np.where(
        (virtual_timestamp > valid_split_time) & \
        (virtual_timestamp < epoch_valid_split_time)
    )[0]
    return train_idx, valid_idx, epoch_valid_idx
