from argparse import ArgumentParser
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import BertConfig

from src.config import Config
from src.data import get_loader
from src.evaluate import predict
from src.models import AktEncoderDecoderModel, SaintEncoderDecoderModel
from src.optim import NoamLR
from src.utils import set_seed, timer
from src.validation import virtual_time_split


def main():
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()

    # settings
    config_path = Path(args.config_file)
    config = Config.load(config_path)

    warnings.filterwarnings('ignore')
    set_seed(config.seed)
    start_time = time.time()
    
    with timer('load data'):
        DATA_DIR = './input/riiid-test-answer-prediction/'
        usecols = [
            'row_id',
            'timestamp',
            'user_id',
            'content_id',
            'content_type_id',
            'answered_correctly',
            'prior_question_elapsed_time', 
        ]
        dtype = {
            'row_id': 'int64',
            'timestamp': 'int64',
            'user_id': 'int32',
            'content_id': 'int16',
            'content_type_id': 'int8',
            'answered_correctly':'int8',
            'prior_question_elapsed_time': 'float32'
        }

        train_df = pd.read_csv(DATA_DIR + 'train.csv', usecols=usecols, dtype=dtype)
        question_df = pd.read_csv(DATA_DIR + 'questions.csv', usecols=['question_id', 'part'])

    train_df = train_df[train_df['content_type_id'] == 0].reset_index(drop=True)

    question_df['part'] += 1  # 0: padding id, 1: start id
    train_df['content_id'] += 2  # 0: padding id, 1: start id
    question_df['question_id'] += 2
    train_df = train_df.merge(question_df, how='left', left_on='content_id', right_on='question_id')

    with timer('validation split'):
        train_idx, valid_idx, epoch_valid_idx = virtual_time_split(train_df,
                                                                   valid_size=config.valid_size,
                                                                   epoch_valid_size=config.epoch_valid_size)
        valid_y = train_df.iloc[valid_idx]['answered_correctly'].values
        epoch_valid_y = train_df.iloc[epoch_valid_idx]['answered_correctly'].values

    print('-' * 20)
    print(f'train size: {len(train_idx)}')
    print(f'valid size: {len(valid_idx)}')

    with timer('prepare data loader'):
        train_loader, valid_loader, epoch_valid_loader = get_loader(config, train_df, train_idx, valid_idx, epoch_valid_idx)

    with timer('train'):
        if config.model == 'akt':
            content_encoder_config = BertConfig(**config.content_encoder_config)
            knowledge_encoder_config = BertConfig(**config.knowledge_encoder_config)
            decoder_config = BertConfig(**config.decoder_config)

            content_encoder_config.max_position_embeddings = config.window_size + 1
            knowledge_encoder_config.max_position_embeddings = config.window_size
            decoder_config.max_position_embeddings = config.window_size + 1

            model = AktEncoderDecoderModel(content_encoder_config, knowledge_encoder_config, decoder_config)

        elif config.model == 'saint':
            encoder_config = BertConfig(**config.encoder_config)
            decoder_config = BertConfig(**config.decoder_config)

            encoder_config.max_position_embeddings = config.window_size
            decoder_config.max_position_embeddings = config.window_size

            model = SaintEncoderDecoderModel(encoder_config, decoder_config)

        else:
            raise ValueError(f'Unknown model: {config.model}')

        model.to(config.device)
        model.zero_grad()
        
        optimizer = optim.Adam(model.parameters(), **config.optimizer_params)
        scheduler = NoamLR(optimizer, warmup_steps=config.warmup_steps)
        loss_ema = None

        for epoch in range(config.n_epochs):
            epoch_start_time = time.time()
            model.train()

            progress = tqdm(train_loader, desc=f'epoch {epoch + 1}', leave=False)
            for i, (x_batch, w_batch, y_batch) in enumerate(progress):
                y_pred = model(**x_batch.to(config.device).to_dict())
                loss = nn.BCEWithLogitsLoss(weight=w_batch.to(config.device))(y_pred, y_batch.to(config.device))
                loss.backward()

                if (
                    config.gradient_accumulation_steps is None
                    or (i + 1) % config.gradient_accumulation_steps == 0
                ):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                
                loss_ema = loss_ema * 0.9 + loss.item() * 0.1 if loss_ema is not None else loss.item()
                progress.set_postfix(loss=loss_ema)

            valid_preds = predict(model, epoch_valid_loader, device=config.device)
            valid_score = roc_auc_score(epoch_valid_y, valid_preds)

            elapsed_time = time.time() - epoch_start_time
            print(f'Epoch {epoch + 1}/{config.n_epochs} \t valid score: {valid_score:.5f} \t time: {elapsed_time / 60:.1f} min')
    
    with timer('predict'):
        valid_preds = predict(model, valid_loader, device=config.device)
        valid_score = roc_auc_score(valid_y, valid_preds)

    print(f'valid score: {valid_score:.5f}')

    output_dir = Path(f'./output/{config_path.stem}/')
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / 'model.pt')
    torch.save(optimizer.state_dict(), output_dir / 'optimizer.pt')

    elapsed_time = time.time() - start_time
    print(f'all processes done in {elapsed_time / 60:.1f} min.')


if __name__ == '__main__':
    main()
