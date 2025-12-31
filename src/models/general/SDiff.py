# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import math

from models.BaseModel import BaseModel


class SDiff(BaseModel):
    reader = 'BaseReader'
    runner = 'SDiffRunner'
    extra_log_args = ['emb_size', 'T', 'K_eig', 'alpha_min', 'sigma_max', 'guidance_s']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=256)
        parser.add_argument('--T', type=int, default=20, help='Diffusion steps for sampling')
        parser.add_argument('--K_eig', type=int, default=100)
        parser.add_argument('--alpha_min', type=float, default=0.001)
        parser.add_argument('--sigma_max', type=float, default=1)
        parser.add_argument('--test_all', type=int, default=0, help='1 for full-ranking, 0 for candidate-based')
        parser.add_argument('--guidance_s', type=float, default=0.2,
                            help='Unconditional weight in sampling. '
                                 's=0.0: fully conditional; s=0.2: 80%% cond + 20%% uncond')
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        self.emb_size = args.emb_size
        self.T = args.T
        self.K = min(args.K_eig, self.item_num)
        self.alpha_min = args.alpha_min
        self.sigma_max = args.sigma_max
        self.test_all = args.test_all
        self.args = args
        self.device = args.device

        self.user_his = self._build_user_history(corpus)
        self._build_item_graph()

        # 使用修正后的 Denoiser（FiLM 正确调制 vt）
        self.denoiser = SDiffDenoiser(self.emb_size, self.K)
        self.time_proj = nn.Linear(self.emb_size, self.emb_size)
        self.apply(self.init_weights)

    def _build_user_history(self, corpus):
        user_his = {}
        for u, items in corpus.train_clicked_set.items():
            user_his[int(u)] = list(items)
        return user_his

    def _build_item_graph(self):
        rows, cols = [], []
        for uid, items in self.user_his.items():
            if items:
                rows.extend([uid] * len(items))
                cols.extend(items)
        if not rows:
            raise ValueError("No interaction data")

        data = np.ones(len(rows), dtype=np.float32)
        X = sp.csr_matrix((data, (rows, cols)), shape=(self.user_num, self.item_num))

        user_deg = np.array(X.sum(1)).flatten()
        item_deg = np.array(X.sum(0)).flatten()
        user_deg[user_deg == 0] = 1
        item_deg[item_deg == 0] = 1

        D_U_inv_sqrt = sp.diags(1.0 / np.sqrt(user_deg))
        D_I_inv_sqrt = sp.diags(1.0 / np.sqrt(item_deg))
        X_norm = D_U_inv_sqrt @ X @ D_I_inv_sqrt
        A = X_norm.T @ X_norm
        L = sp.eye(A.shape[0], format='csr') - A

        max_k = min(self.K, A.shape[0] - 1)
        if max_k < 1:
            max_k = min(50, self.item_num)

        try:
            eigenvals, eigenvecs = eigsh(L, k=max_k, which='SM')
        except Exception as e:
            print(f"eigsh failed: {e}, using random orthogonal basis")
            eigenvals = np.zeros(max_k)
            eigenvecs = np.random.randn(self.item_num, max_k)
            eigenvecs, _ = np.linalg.qr(eigenvecs)

        idx = np.argsort(eigenvals)[:max_k]
        self.eigenvals = torch.FloatTensor(eigenvals[idx]).to(self.device)
        self.U = torch.FloatTensor(eigenvecs[:, idx]).to(self.device)
        self.U = torch.nn.functional.normalize(self.U, dim=0, p=2)

    def sinusoidal_time_embedding(self, t, dim):
        device = t.device
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=device) / half_dim)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, feed_dict):
        user_ids = feed_dict['user_id'].view(-1)  # [B]
        B = user_ids.shape[0]

        x0 = torch.zeros(B, self.item_num, device=self.device)
        for i, uid in enumerate(user_ids.cpu().tolist()):
            uid = int(uid)
            if uid in self.user_his:
                items_tensor = torch.LongTensor(self.user_his[uid]).to(self.device)
                x0[i].scatter_(0, items_tensor, 1.0)

        if self.training:
            mask = (torch.rand_like(x0) < 0.5).float()
            c_spatial = x0 * mask
            c_spectral = c_spatial @ self.U  # [B, K]

            p_uncond = self.args.guidance_s
            if torch.rand(1).item() < p_uncond:
                c_spectral = torch.zeros_like(c_spectral)

            t_k = torch.randint(1, self.T + 1, (B,), device=self.device)
            t_norm = t_k.float() / self.T

            lam_t = torch.exp(-t_norm.unsqueeze(1) * self.eigenvals.unsqueeze(0))
            alpha_t = lam_t
            sigma_t = torch.sqrt(1 - alpha_t ** 2).clamp(max=self.sigma_max)

            v0 = x0 @ self.U
            eps = torch.randn_like(v0)
            vt = alpha_t * v0 + sigma_t * eps

            time_embed = self.sinusoidal_time_embedding(t_norm, self.emb_size)
            time_feat = self.time_proj(time_embed)
            v0_pred = self.denoiser(vt, c_spectral, time_feat)

            loss = torch.nn.functional.mse_loss(v0_pred, v0)
            dummy_pred = torch.zeros(B, self.item_num, device=self.device)
            return {'loss': loss, 'prediction': dummy_pred}

        else:
            c_spatial = x0.clone()
            c_spectral = c_spatial @ self.U

            v_curr = torch.randn(B, self.K, device=self.device)
            s = getattr(self.args, 'guidance_s', 0.2)

            for step in reversed(range(1, self.T + 1)):
                t_val = torch.full((B,), step / self.T, device=self.device)

                lam_t = torch.exp(-t_val.unsqueeze(1) * self.eigenvals.unsqueeze(0))
                alpha_t = lam_t
                sigma_t = torch.sqrt(1 - alpha_t ** 2).clamp(max=self.sigma_max)

                time_embed = self.sinusoidal_time_embedding(t_val, self.emb_size)
                time_feat = self.time_proj(time_embed)

                if s == 0.0:
                    v0_pred = self.denoiser(v_curr, c_spectral, time_feat)
                else:
                    pred_cond = self.denoiser(v_curr, c_spectral, time_feat)
                    pred_uncond = self.denoiser(v_curr, torch.zeros_like(c_spectral), time_feat)
                    v0_pred = (1 - s) * pred_cond + s * pred_uncond

                if step > 1:
                    t_prev = torch.full((B,), (step - 1) / self.T, device=self.device)
                    lam_prev = torch.exp(-t_prev.unsqueeze(1) * self.eigenvals.unsqueeze(0))
                    alpha_prev = lam_prev
                    sigma_prev = torch.sqrt(1 - lam_prev ** 2).clamp(max=self.sigma_max)
                    noise = torch.randn_like(v_curr)
                    v_curr = alpha_prev * v0_pred + sigma_prev * noise
                else:
                    v_curr = v0_pred

            full_pred = v_curr @ self.U.T
            full_pred = full_pred * (1 - x0)

            if 'candidates' in feed_dict:
                candidates = feed_dict['candidates']
                scores = torch.gather(full_pred, 1, candidates)
            elif 'item_id' in feed_dict:
                candidates = feed_dict['item_id']
                scores = torch.gather(full_pred, 1, candidates)
            else:
                scores = full_pred

            return {'prediction': scores}

    class Dataset(BaseModel.Dataset):
        def _get_feed_dict(self, index: int) -> dict:
            feed_dict = {
                'user_id': np.array([self.data['user_id'][index]], dtype=np.int64)
            }
            if 'item_id' in self.data and 'neg_items' in self.data:
                pos = self.data['item_id'][index]
                negs = self.data['neg_items'][index]
                candidates = [pos] + negs
                feed_dict['item_id'] = np.array(candidates, dtype=np.int64)
            elif 'item_id' in self.data:
                feed_dict['item_id'] = np.array([self.data['item_id'][index]], dtype=np.int64)
            return feed_dict


# ==================== 修正后的 Denoiser：FiLM 正确调制 vt ====================
class SDiffDenoiser(nn.Module):
    def __init__(self, emb_size, K):
        super().__init__()
        # FiLM 参数生成网络：由 c_spectral（用户条件）生成 gamma 和 beta
        self.gamma_net = nn.Sequential(
            nn.Linear(K, 128),
            nn.ReLU(),
            nn.Linear(128, K)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(K, 128),
            nn.ReLU(),
            nn.Linear(128, K)
        )
        # 主干 MLP：输入为 [modulated_vt; time_feat]
        self.mlp = nn.Sequential(
            nn.Linear(K + emb_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, K)
        )

    def forward(self, vt, c_spectral, time_feat):
        # 用用户历史条件 c_spectral 生成 FiLM 参数
        gamma = self.gamma_net(c_spectral)   # [B, K]
        beta = self.beta_net(c_spectral)     # [B, K]
        
        # 对主干特征 vt 进行 FiLM 调制
        vt_modulated = gamma * vt + beta     # [B, K]
        
        # 拼接时间特征（控制去噪强度）
        combined = torch.cat([vt_modulated, time_feat], dim=-1)  # [B, K + emb_size]
        
        return self.mlp(combined)