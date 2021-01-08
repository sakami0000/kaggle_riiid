import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def predict(model: nn.Module,
            data_loader: DataLoader,
            device: torch.device = torch.device('cuda')) -> np.ndarray:
    model.eval()
    preds = []
    
    with torch.no_grad():
        for x_batch in tqdm(data_loader, desc='predict', leave=False):
            gather_idx = (x_batch.decoder_attention_mask.sum(dim=1, keepdim=True) - 1).long().to(device)
            y_pred = model(**x_batch.to(device).to_dict()).gather(1, gather_idx).sigmoid().detach().cpu()
            preds.append(y_pred)
            
    return torch.cat(preds, dim=0).numpy()
