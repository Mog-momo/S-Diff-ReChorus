# -*- coding: UTF-8 -*-
# @Author  : Your Name
# @Email   : your@email.com

from .BaseRunner import BaseRunner
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from utils import utils


class SDiffRunner(BaseRunner):
    """
    Runner for Spectral Diffusion (SDiff) model.
    - Training uses diffusion loss returned in `forward()`.
    - Evaluation uses standard full-ranking or candidate-based ranking from parent `BaseRunner.evaluate()`.
    """

    def fit(self, dataset, epoch=-1) -> float:
        """
        Train one epoch.
        Note: No negative sampling; loss is computed inside model.forward().
        """
        model = dataset.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)

        # Optional: pre-epoch logic (e.g., negative sampling), but SDiff doesn't need it
        if hasattr(dataset, 'actions_before_epoch'):
            dataset.actions_before_epoch()

        model.train()
        loss_lst = []
        dl = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_batch,
            pin_memory=self.pin_memory
        )

        for batch in tqdm(dl, leave=False, desc=f'Epoch {epoch:<3}', ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)
            model.optimizer.zero_grad()
            out_dict = model(batch)
            loss = out_dict['loss']  # Must be a scalar tensor
            loss.backward()
            model.optimizer.step()
            loss_lst.append(loss.item())

        return float(np.mean(loss_lst))

    # ⚠️ Do NOT override `evaluate()` unless necessary.
    # The parent `BaseRunner.evaluate()` already handles:
    #   - full-ranking vs candidate-based
    #   - HR/NDCG computation
    #   - calling model(feed_dict) and using 'prediction'