import copy
import torch
import torch.nn.functional as F


class Rollout(object):
    def __init__(self, generator, beta):
        self.origin_generator = generator
        self.own_generator = copy.deepcopy(generator)
        self.beta = beta

    def update_params(self):
        dic = {}
        for name, param in self.origin_generator.named_parameters():
            dic[name] = param.data

        for name, param in self.own_generator.named_parameters():
            if 'embedding' in name:
                param.data = dic[name]
            else:
                param.data = self.beta * param.data + (1.0 - self.beta) * dic[name]

    def get_reward(self, whole_traj, whole_len, num_Monte_Carlo, discriminator):
        batch_size, max_seq_len = whole_traj.size()
        rewards = torch.zeros((batch_size, max_seq_len-1), dtype=torch.float32, device=whole_traj.device)

        for Monte_Carlo_id in range(num_Monte_Carlo):
            for i in range(1, max_seq_len):
                cur_traj = whole_traj[:, :i+1].clone()
                traj_len = whole_len.clone()
                completed_trajectory = self.own_generator.complete_trajectory(cur_traj, traj_len)
                batch_mask = (i < traj_len)
                with torch.no_grad():
                    logits = discriminator(completed_trajectory[batch_mask], whole_len[batch_mask])
                prob = F.softmax(logits, dim=1)
                rewards[batch_mask, i-1] += prob[:, 1]

        rewards = rewards / num_Monte_Carlo
        return rewards
