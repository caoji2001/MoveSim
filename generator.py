import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, config, start_rid_list, start_rid_probs, len_list, len_probs, device, mat1, mat2, mat3=None):
        super(Generator, self).__init__()

        self.rid_emb_dim = config['rid_emb_dim']
        self.time_emb_dim = config['time_emb_dim']
        self.embed_dim = config['rid_emb_dim'] + config['time_emb_dim']
        self.hidden_dim = config['hidden_dim']

        self.num_rid = config['num_rid']
        self.num_time = config['num_time']

        self.start_rid_list = start_rid_list
        self.start_rid_probs = start_rid_probs
        self.len_list = len_list
        self.len_probs = len_probs

        self.device = device

        self.mat1 = torch.from_numpy(np.concatenate([mat1, np.zeros((1, mat1.shape[1]), dtype=np.float32)], axis=0)).to(self.device)
        self.mat2 = torch.from_numpy(np.concatenate([mat2, np.zeros((1, mat2.shape[1]), dtype=np.float32)], axis=0)).to(self.device)
        # Corresponding to Function Similarity Matrix
        self.mat3 = None
        if mat3 is not None:
            self.mat3 = torch.from_numpy(np.concatenate([mat3, np.zeros(mat3.shape[1], dtype=np.float32)], axis=0)).to(self.device)

        self.rid_embedding = nn.Embedding(self.num_rid+1, self.rid_emb_dim, self.num_rid)
        self.time_embedding = nn.Embedding(self.num_time+1, self.time_emb_dim, self.num_time)

        self.q_proj_1 = nn.Linear(self.embed_dim, self.hidden_dim)
        self.k_proj_1 = nn.Linear(self.embed_dim, self.hidden_dim)
        self.v_proj_1 = nn.Linear(self.embed_dim, self.hidden_dim)
        self.attn_1 = nn.MultiheadAttention(self.hidden_dim, 4, batch_first=True)

        self.q_proj_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attn_2 = nn.MultiheadAttention(self.hidden_dim, 1, batch_first=True)

        self.linear = nn.Linear(self.hidden_dim, self.num_rid)

        self.linear_m1_1 = nn.Linear(self.num_rid, self.hidden_dim)
        self.linear_m1_2 = nn.Linear(self.hidden_dim, self.num_rid)

        self.linear_m2_1 = nn.Linear(self.num_rid, self.hidden_dim)
        self.linear_m2_2 = nn.Linear(self.hidden_dim, self.num_rid)

        if mat3 is not None:
            self.linear_m3_1 = nn.Linear(self.num_rid, self.hidden_dim)
            self.linear_m3_2 = nn.Linear(self.hidden_dim, self.num_rid)

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def forward(self, input_traj, input_time, input_len):
        """
        Parameters:
        input_traj (tensor): (batch_size, max_seq_len)
        input_time (tensor): (batch_size, max_seq_len)
        input_len (tensor): (batch_size, )

        Return:
        logits (tensor): (batch_size, max_seq_len, num_rid)
        """
        input_traj_emb = self.rid_embedding(input_traj)
        input_time_emb = self.time_embedding(input_time)
        x = torch.cat([input_traj_emb, input_time_emb], dim=2)

        mask = torch.arange(input_traj.size(1), dtype=torch.int64, device=input_len.device).unsqueeze(0) < input_len.unsqueeze(1)
        padding_mask = ~mask

        q = self.q_proj_1(x)
        q = F.relu(q)
        k = self.k_proj_1(x)
        k = F.relu(k)
        v = self.v_proj_1(x)
        v = F.relu(v)
        x, _ = self.attn_1(q, k, v, key_padding_mask=padding_mask)

        q = self.q_proj_2(x)
        q = F.relu(q)
        k = self.k_proj_2(x)
        k = F.relu(k)
        v = self.v_proj_2(x)
        v = F.relu(v)
        x, _ = self.attn_2(q, k, v, key_padding_mask=padding_mask)

        x = self.linear(x)

        mat1_info = self.mat1[input_traj]
        mat1_info = self.linear_m1_1(mat1_info)
        mat1_info = F.relu(mat1_info)
        mat1_info = self.linear_m1_2(mat1_info)
        mat1_info = F.sigmoid(mat1_info)
        mat1_info = F.normalize(mat1_info, dim=-1)

        mat2_info = self.mat2[input_traj]
        mat2_info = self.linear_m2_1(mat2_info)
        mat2_info = F.relu(mat2_info)
        mat2_info = self.linear_m2_2(mat2_info)
        mat2_info = F.sigmoid(mat2_info)
        mat2_info = F.normalize(mat2_info, dim=-1)

        if self.mat3 is not None:
            mat3_info = self.mat3[input_traj]
            mat3_info = self.linear_m3_1(mat3_info)
            mat3_info = F.relu(mat3_info)
            mat3_info = self.linear_m3_2(mat3_info)
            mat3_info = F.sigmoid(mat3_info)
            mat3_info = F.normalize(mat3_info, dim=-1)

        logits = x + torch.mul(x, mat1_info) + torch.mul(x, mat2_info)
        if self.mat3 is not None:
            logits += torch.mul(x, mat3_info)

        return logits
    
    def sample(self, num_samples):
        """
        Parameters:
        num_samples (int)

        Returns:
        result (list)
        """
        start_info = np.random.choice(self.start_rid_list, size=num_samples, p=self.start_rid_probs)

        length = np.random.choice(self.len_list, size=num_samples, p=self.len_probs)
        max_len = np.max(length)

        samples = torch.zeros((num_samples, max_len), dtype=torch.int64, device=self.device)
        samples[:, 0] = torch.from_numpy(start_info)

        batch_size = 32
        with torch.no_grad():
            for batch_id in tqdm(range(math.ceil(num_samples/batch_size)), ncols=100, desc='generating samples'):
                batch_idx_start = batch_id * batch_size
                batch_idx_end = min((batch_id + 1) * batch_size, num_samples)
                for i in range(1, max_len):
                    logits = self.forward(
                        input_traj=samples[batch_idx_start:batch_idx_end, :i],
                        input_time=torch.arange(i, dtype=torch.int64, device=self.device).unsqueeze(0).expand(batch_idx_end-batch_idx_start, -1),
                        input_len=torch.tensor([i], dtype=torch.int64, device=self.device).expand(batch_idx_end-batch_idx_start)
                    )
                    pred = torch.multinomial(F.softmax(logits[:, -1, :], dim=1), 1).squeeze(1)
                    samples[batch_idx_start:batch_idx_end, i] = pred

        result = []
        for i, sample in enumerate(samples):
            result.append(sample[:length[i]].tolist())

        return result

    def complete_trajectory(self, cur_traj, traj_len):
        """
        Parameters:
        cur_traj (tensor): (batch_size, given_len)
        traj_len (tensor): (batch_size, )

        Returns:
        completed_trajectory (tensor): (batch_size, max_seq_len)
        """
        batch_size, given_len = cur_traj.size()
        max_seq_len = torch.max(traj_len).item()
        completed_trajectory = torch.cat([
            cur_traj,
            torch.full((batch_size, max_seq_len-given_len), fill_value=self.num_rid, dtype=torch.int64, device=self.device)
        ], dim=1)

        for i in range(given_len, max_seq_len):
            batch_mask = (i < traj_len)
            batch_mask_sum = torch.sum(batch_mask).item()
            assert batch_mask_sum > 0

            with torch.no_grad():
                logits = self.forward(
                    input_traj=completed_trajectory[batch_mask, :i],
                    input_time=torch.arange(i, dtype=torch.int64, device=self.device).unsqueeze(0).expand(batch_mask_sum, -1),
                    input_len=torch.full((batch_mask_sum, ), i, dtype=torch.int64, device=self.device)
                )
            pred = torch.multinomial(F.softmax(logits[:, -1, :], dim=1), 1).squeeze(1)
            completed_trajectory[batch_mask, i] = pred

        return completed_trajectory
