from typing import Dict, List, Tuple, NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .config import Config


class Example(NamedTuple):
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    timestamp: torch.LongTensor
    category_ids: torch.LongTensor
    elapsed_time: torch.FloatTensor
    response_ids: torch.LongTensor
    decoder_attention_mask: torch.LongTensor

    def to(self, device: torch.device) -> 'Example':
        return Example(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            timestamp=self.timestamp.to(device),
            category_ids=self.category_ids.to(device),
            elapsed_time=self.elapsed_time.to(device),
            response_ids=self.response_ids.to(device),
            decoder_attention_mask=self.decoder_attention_mask.to(device)
        )

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            timestamp=self.timestamp,
            category_ids=self.category_ids,
            elapsed_time=self.elapsed_time,
            response_ids=self.response_ids,
            decoder_attention_mask=self.decoder_attention_mask
        )


def get_user_sequences(df: pd.DataFrame) -> Dict[int, Dict[str, List[int]]]:
    df_sorted = df.sort_values(['timestamp', 'row_id'])
    df_sorted['elapsed_time'] = df['prior_question_elapsed_time'].fillna(0) / 1000  # seconds

    user_seqs = (
        df_sorted
        .groupby('user_id')
        .agg({
            'content_id': list,
            'part': list,
            'timestamp': list,
            'elapsed_time': list,
            'answered_correctly': list
        })
        .to_dict(orient='index')
    )
    return user_seqs


class TrainDataset(Dataset):
    
    def __init__(self,
                 train_user_seqs: Dict[int, Dict[str, List[int]]],
                 window_size: int = 100,
                 stride_size: int = 100):
        self.train_user_seqs = train_user_seqs
        self.window_size = window_size
        self.stride_size = stride_size

        self.all_examples = []
        self.all_target_masks = []
        self.all_targets = []
        for user_seq in tqdm(train_user_seqs.values(), desc='prepare train seqs'):
            sequence_length = len(user_seq['content_id'])
            
            for start_idx in range(0, sequence_length, stride_size):
                end_idx = start_idx + window_size

                if start_idx > 0 and end_idx >= sequence_length:
                    start_idx = max(0, sequence_length - window_size)

                    if user_seq['content_id'][start_idx:end_idx] == content_id:
                        # same as previous example
                        continue
                
                # target -> used for loss calculation
                #   0: incorrect answer, padding id
                #   1: correct answer

                # target_for_input -> used for response embedding
                #   0: padding id
                #   1: start id
                #   2: incorrect answer
                #   3: correct answer

                content_id = user_seq['content_id'][start_idx:end_idx]
                part = user_seq['part'][start_idx:end_idx]
                timestamp = user_seq['timestamp'][start_idx:end_idx]
                elapsed_time = user_seq['elapsed_time'][start_idx:end_idx]
                target = user_seq['answered_correctly'][start_idx:end_idx]
                target_for_input = [1] + [label + 2 for label in target[:-1]]  # add start id

                if len(content_id) < self.window_size:
                    # padding
                    pad_size = self.window_size - len(content_id)
                    content_id += [0] * pad_size
                    part += [0] * pad_size
                    timestamp += [0] * pad_size
                    elapsed_time += [0] * pad_size
                    target += [0] * pad_size
                    target_for_input += [0] * pad_size

                input_ids = torch.LongTensor(content_id)
                attention_mask = (input_ids > 0).float()
                timestamp = torch.LongTensor(timestamp)
                category_ids = torch.LongTensor(part)
                elapsed_time = torch.FloatTensor(elapsed_time)
                response_ids = torch.LongTensor(target_for_input)
                decoder_attention_mask = (response_ids > 0).float()
                
                example = Example(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    timestamp=timestamp,
                    category_ids=category_ids,
                    elapsed_time=elapsed_time,
                    response_ids=response_ids,
                    decoder_attention_mask=decoder_attention_mask
                )

                self.all_examples.append(example)
                self.all_target_masks.append(decoder_attention_mask)
                self.all_targets.append(torch.FloatTensor(target))
        
    def __len__(self) -> int:
        return len(self.all_examples)
    
    def __getitem__(self, idx: int) -> Tuple[Example, torch.Tensor, torch.Tensor]:
        return self.all_examples[idx], self.all_target_masks[idx], self.all_targets[idx]


class ValidDataset(Dataset):
    
    def __init__(self,
                 train_df: pd.DataFrame,
                 train_user_seqs: Dict[int, Dict[str, List[int]]],
                 valid_user_seqs: Dict[int, Dict[str, List[int]]],
                 valid_idx: np.ndarray,
                 window_size: int):
        self.train_user_seqs = train_user_seqs
        self.valid_user_seqs = valid_user_seqs
        self.valid_idx = valid_idx
        self.valid_idx_map = self._get_valid_idx_map(train_df, valid_idx)
        self.window_size = window_size

    @staticmethod
    def _get_valid_idx_map(
        train_df: pd.DataFrame,
        valid_idx: np.ndarray
    ) -> Dict[int, Dict[str, int]]:
        valid_df = (
            train_df
            .reset_index(drop=True)
            .iloc[valid_idx]
            .sort_values(['timestamp', 'row_id'])
        )
        valid_df['list_idx'] = valid_df.groupby('user_id').cumcount() + 1
        valid_idx_map = valid_df[['user_id', 'list_idx']].to_dict(orient='index')
        return valid_idx_map

    def __len__(self) -> int:
        return len(self.valid_idx)
    
    def get_last_slice(self, idx: int) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
        idx_map = self.valid_idx_map[self.valid_idx[idx]]
        user_id = idx_map['user_id']
        list_idx = idx_map['list_idx']

        train_content_id = []
        train_part = []
        train_timestamp = []
        train_elapsed_time = []
        train_target = []
        
        if user_id in self.train_user_seqs and self.window_size > list_idx:
            train_sequence_length = self.window_size - list_idx

            train_content_id = self.train_user_seqs[user_id]['content_id'][-train_sequence_length:]
            train_part = self.train_user_seqs[user_id]['part'][-train_sequence_length:]
            train_timestamp = self.train_user_seqs[user_id]['timestamp'][-train_sequence_length:]
            train_elapsed_time = self.train_user_seqs[user_id]['elapsed_time'][-train_sequence_length:]
            train_target = self.train_user_seqs[user_id]['answered_correctly'][-train_sequence_length:]
            
        start_idx = max(list_idx - self.window_size, 0)
        valid_content_id = self.valid_user_seqs[user_id]['content_id'][start_idx:list_idx]
        valid_part = self.valid_user_seqs[user_id]['part'][start_idx:list_idx]
        valid_timestamp = self.valid_user_seqs[user_id]['timestamp'][start_idx:list_idx]
        valid_elapsed_time = self.valid_user_seqs[user_id]['elapsed_time'][start_idx:list_idx]
        valid_target = self.valid_user_seqs[user_id]['answered_correctly'][start_idx:list_idx]

        # drop simultaneously occured samples
        same_timestamp_count = valid_timestamp.count(valid_timestamp[-1])
        if same_timestamp_count >= 2:
            valid_elapsed_time = valid_elapsed_time[:-(same_timestamp_count - 1)]

        # target -> used for response embedding
        #   0: padding id
        #   1: start id
        #   2: incorrect answer
        #   3: correct answer
        content_id = train_content_id + valid_content_id[:-same_timestamp_count] + [valid_content_id[-1]]
        part = train_part + valid_part[:-same_timestamp_count] + [valid_part[-1]]
        timestamp = train_timestamp + valid_timestamp[:-same_timestamp_count] + [valid_timestamp[-1]]
        elapsed_time = train_elapsed_time + valid_elapsed_time
        target = [1] + [label + 2 for label in train_target + valid_target[:-same_timestamp_count]]  # add start id
        
        sequence_length = len(content_id)
        if sequence_length < self.window_size:
            # padding
            pad_size = self.window_size - sequence_length
            content_id += [0] * pad_size
            part += [0] * pad_size
            timestamp += [-1] * pad_size
            elapsed_time += [0] * pad_size
            target += [0] * pad_size
        
        return content_id, part, timestamp, elapsed_time, target

    def __getitem__(self, idx: int) -> Example:
        content_id, part, timestamp, elapsed_time, target = self.get_last_slice(idx)
        
        input_ids = torch.LongTensor(content_id)
        attention_mask = (input_ids > 0).float()
        timestamp = torch.LongTensor(timestamp)
        category_ids = torch.LongTensor(part)
        elapsed_time = torch.FloatTensor(elapsed_time)
        response_ids = torch.LongTensor(target)
        decoder_attention_mask = (response_ids > 0).float()
        
        example = Example(
            input_ids=input_ids,
            attention_mask=attention_mask,
            timestamp=timestamp,
            category_ids=category_ids,
            elapsed_time=elapsed_time,
            response_ids=response_ids,
            decoder_attention_mask=decoder_attention_mask
        )
        return example
