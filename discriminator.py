import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.embedding_dim = config['discriminator_rid_emb_dim']
        self.num_rid = config['num_rid']
        self.dropout_p = config['discriminator_dropout_p']

        self.embedding = nn.Embedding(self.num_rid+1, self.embedding_dim, self.num_rid)

        self.num_filters = [100, 200, 200, 200]
        self.filter_sizes = [1, 2, 3, 4]

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filter, (filter_size, self.embedding_dim)) for num_filter, filter_size in zip(self.num_filters, self.filter_sizes)]
        )

        self.highway = nn.Linear(sum(self.num_filters), sum(self.num_filters))
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.linear = nn.Linear(sum(self.num_filters), 2)

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def forward(self, input_traj, input_len):
        """
        Parameters:
        input_traj (tensor): (batch_size, seq_len)
        input_len (tensor): (batch_size, )

        Return:
        pred (tensor): (batch_size, 2)
        """
        assert torch.min(input_len).item() >= max(self.filter_sizes)
        emb = self.embedding(input_traj)

        emb = emb.unsqueeze(1)
        conv_result = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pool_result = []
        for i, data in enumerate(conv_result):
            mask = torch.arange(input_traj.size(1)-self.filter_sizes[i]+1, dtype=torch.int64, device=input_traj.device).unsqueeze(0) < (input_len-self.filter_sizes[i]+1).unsqueeze(1)
            mask = mask.unsqueeze(1)
            mask = mask.expand_as(data)
            data = data.masked_fill(~mask, float('-inf'))
            pool_result.append(F.max_pool1d(data, data.size(2)).squeeze(2))

        pred = torch.cat(pool_result, 1)
        highway = self.highway(pred)
        pred = F.sigmoid(highway) * F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred = self.dropout(pred)
        pred = self.linear(pred)

        return pred
