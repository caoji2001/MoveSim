import os
import copy
import argparse
import random
from collections import Counter
from tqdm import tqdm
import yaml
import numpy as np
import pandas as pd
from shapely.geometry import LineString
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import set_seed, haversine
from dataset import GeneratorDataset, DiscriminatorDataset
from generator import Generator
from discriminator import Discriminator
from rollout import Rollout


def generator_pretrain_collate_fn(items):
    input_traj = []
    input_time = []
    input_len = []
    label_traj = []

    for item in items:
        input_traj.append(copy.deepcopy(item[0][:-1]))
        input_time.append(copy.deepcopy(item[1][:-1]))
        input_len.append(len(item[0]) - 1)
        label_traj.append(copy.deepcopy(item[0][1:]))

    max_input_len = max(input_len)
    for i in range(len(input_traj)):
        padding_len = max_input_len - input_len[i]
        input_traj[i].extend([config['num_rid']] * padding_len)
        input_time[i].extend([config['num_time']] * padding_len)
        label_traj[i].extend([config['num_rid']] * padding_len)

    input_traj = torch.tensor(input_traj, dtype=torch.int64)
    input_time = torch.tensor(input_time, dtype=torch.int64)
    input_len = torch.tensor(input_len, dtype=torch.int64)
    label_traj = torch.tensor(label_traj, dtype=torch.int64)

    return input_traj, input_time, input_len, label_traj

def discriminator_collate_fn(items):
    input_traj = []
    input_len = []
    label = []

    for item in items:
        input_traj.append(copy.deepcopy(item[0]))
        input_len.append(len(item[0]))
        label.append(copy.deepcopy(item[1]))

    max_input_len = max(input_len)
    for i in range(len(input_traj)):
        padding_len = max_input_len - input_len[i]
        input_traj[i].extend([config['num_rid']] * padding_len)

    input_traj = torch.tensor(input_traj, dtype=torch.int64)
    input_len = torch.tensor(input_len, dtype=torch.int64)
    label = torch.tensor(label, dtype=torch.int64)

    return input_traj, input_len, label

def generator_adversarial_collate_fn(items):
    input_traj = []
    input_time = []
    input_len = []
    label_traj = []
    whole_traj = []
    whole_time = []
    whole_len = []

    for item in items:
        input_traj.append(copy.deepcopy(item[0][:-1]))
        input_time.append(copy.deepcopy(item[1][:-1]))
        input_len.append(len(item[0]) - 1)
        label_traj.append(copy.deepcopy(item[0][1:]))
        whole_traj.append(copy.deepcopy(item[0]))
        whole_time.append(copy.deepcopy(item[1]))
        whole_len.append(len(item[0]))

    max_input_len = max(input_len)
    for i in range(len(input_traj)):
        padding_len = max_input_len - input_len[i]
        input_traj[i].extend([config['num_rid']] * padding_len)
        input_time[i].extend([config['num_time']] * padding_len)
        label_traj[i].extend([config['num_rid']] * padding_len)

    max_whole_len = max(whole_len)
    for i in range(len(whole_traj)):
        padding_len = max_whole_len - whole_len[i]
        whole_traj[i].extend([config['num_rid']] * padding_len)
        whole_time[i].extend([config['num_time']] * padding_len)

    input_traj = torch.tensor(input_traj, dtype=torch.int64)
    input_time = torch.tensor(input_time, dtype=torch.int64)
    input_len = torch.tensor(input_len, dtype=torch.int64)
    label_traj = torch.tensor(label_traj, dtype=torch.int64)

    whole_traj = torch.tensor(whole_traj, dtype=torch.int64)
    whole_time = torch.tensor(whole_time, dtype=torch.int64)
    whole_len = torch.tensor(whole_len, dtype=torch.int64)

    return input_traj, input_time, input_len, label_traj, whole_traj, whole_time, whole_len


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BJ_Taxi')

    parser.add_argument('--pretrain_g_epoch', type=int, default=100)
    parser.add_argument('--pretrain_d_epoch', type=int, default=20)
    parser.add_argument('--adv_epoch', type=int, default=30)
    parser.add_argument('--num_Monte_Carlo', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tensorboard', action='store_true')
    args = parser.parse_args()

    dataset = args.dataset
    device = f'cuda:{args.cuda}'
    seed = args.seed
    set_seed(seed)

    with open(f'./config/{dataset}.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if args.tensorboard:
        os.makedirs(f'./logs/{dataset}_seed{seed}', exist_ok=True)
        writer = SummaryWriter(f'./logs/{dataset}_seed{seed}')

    geo = pd.read_csv(f'./data/{dataset}/roadmap.cleaned.geo')
    road_gps = []
    for _, row in geo.iterrows():
        geo_coordinates = eval(row['coordinates'])
        center = LineString(geo_coordinates).centroid
        road_gps.append((center.x, center.y))
    road_gps = np.array(road_gps, dtype=np.float32)

    real_train_traj = pd.read_csv(f'./data/{dataset}/traj_tra_10000_random.cleaned.csv')
    real_train_traj['rid_list'] = real_train_traj['rid_list'].apply(eval).apply(list)

    """
    Construct matrix info.
    mat1: Physical Distance Matrix
    mat2: Historical Transition Matrix
    Here, we only consider mat1 and mat2, and temporarily do not consider mat3 (i.e., the Function Similarity Matrix mentioned in the text).
    """

    mat1 = np.empty((config['num_rid'], config['num_rid']), dtype=np.float32)
    for rid in tqdm(range(config['num_rid']), ncols=100, desc='constructing physical distance matrix'):
        mat1[rid] = haversine(road_gps[rid], road_gps)
    mat1 /= (1e-8 + np.max(mat1, axis=1, keepdims=True))

    mat2 = np.zeros((config['num_rid'], config['num_rid']), dtype=np.float32)
    for _, row in tqdm(real_train_traj.iterrows(), ncols=100, total=len(real_train_traj), desc='constructing historical transition matrix'):
        rid_list = row['rid_list']
        for prev_rid, next_rid in zip(rid_list[:-1], rid_list[1:]):
            mat2[prev_rid, next_rid] += 1.0
    mat2 /= (1e-8 + np.max(mat2, axis=1, keepdims=True))

    """
    Construct start rid list and corresponding probabilities.
    """

    counter = Counter([row['rid_list'][0] for _, row in real_train_traj.iterrows()])
    start_rid_list = np.array(list(counter.keys()))
    start_rid_probs = np.array(list(counter.values()))
    start_rid_probs = start_rid_probs / np.sum(start_rid_probs)

    """
    Construct trajectory length list and corresponding probabilities.
    """
    counter = Counter([len(row['rid_list']) for _, row in real_train_traj.iterrows()])
    len_list = np.array(list(counter.keys()))
    len_probs = np.array(list(counter.values()))
    len_probs = len_probs / np.sum(len_probs)

    """
    Pretrain Generator
    """

    generator_dataset_train = GeneratorDataset(real_train_traj['rid_list'].tolist())
    generator_dataloader_train = DataLoader(generator_dataset_train, batch_size=32, shuffle=True, collate_fn=generator_pretrain_collate_fn, num_workers=4, pin_memory=True)

    generator = Generator(config, start_rid_list, start_rid_probs, len_list, len_probs, device, mat1, mat2).to(device)
    generator_criterion = nn.CrossEntropyLoss(ignore_index=config['num_rid'])
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    os.makedirs(f'./save/{dataset}', exist_ok=True)

    generator.train()
    for epoch_id in range(args.pretrain_g_epoch):
        loss_sum, loss_cnt = 0, 0
        for batch_id, (input_traj, input_time, input_len, label_traj) in enumerate(tqdm(generator_dataloader_train, ncols=100, desc='pretraining generator')):
            input_traj = input_traj.to(device)
            input_time = input_time.to(device)
            input_len = input_len.to(device)
            label_traj = label_traj.to(device)

            logits = generator(input_traj, input_time, input_len)
            loss = generator_criterion(logits.transpose(1, 2), label_traj)
            loss_sum += loss.item()
            loss_cnt += 1

            loss.backward()
            generator_optimizer.step()
            generator_optimizer.zero_grad()

            if args.tensorboard:
                writer.add_scalar('pretrain_g_loss', loss, len(generator_dataloader_train) * epoch_id + batch_id)

        torch.save(generator.state_dict(), f'./save/{dataset}/pretrained_generator_epoch{epoch_id}.pth')
        print(f'Pretrain Generator, epoch: {epoch_id}, avg_loss: {loss_sum/loss_cnt}')

    """
    Pretrain Discriminator
    """

    fake_rid_list_data = []
    for _, row in real_train_traj.iterrows():
        rid_list = copy.deepcopy(row['rid_list'])
        if random.random() < 0.5:
            random.shuffle(rid_list)
            fake_rid_list_data.append(rid_list)
        else:
            id = np.random.randint(0, len(rid_list))
            rid_list[id] = np.argmax(mat1[rid_list[id]])
            fake_rid_list_data.append(rid_list)

    discriminator_dataset_train = DiscriminatorDataset(real_train_traj['rid_list'].tolist(), fake_rid_list_data)
    discriminator_dataloader_train = DataLoader(discriminator_dataset_train, batch_size=32, shuffle=True, collate_fn=discriminator_collate_fn, num_workers=4, pin_memory=True)

    discriminator = Discriminator(config).to(device)
    discriminator_criterion = nn.CrossEntropyLoss()
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    discriminator.train()
    for epoch_id in range(args.pretrain_d_epoch):
        loss_sum, loss_cnt = 0, 0
        for batch_id, (input_traj, input_len, label) in enumerate(tqdm(discriminator_dataloader_train, ncols=100, desc='pretraining discriminator')):
            input_traj = input_traj.to(device)
            input_len = input_len.to(device)
            label = label.to(device)

            logits = discriminator(input_traj, input_len)
            loss = discriminator_criterion(logits, label)
            loss_sum += loss.item()
            loss_cnt += 1

            loss.backward()
            discriminator_optimizer.step()
            discriminator_optimizer.zero_grad()

            if args.tensorboard:
                writer.add_scalar('pretrain_d_loss', loss, len(discriminator_dataloader_train) * epoch_id + batch_id)

        torch.save(discriminator.state_dict(), f'./save/{dataset}/pretrained_discriminator_epoch{epoch_id}.pth')
        print(f'Pretrain Discriminator, epoch: {epoch_id}, avg_loss: {loss_sum/loss_cnt}')

    """
    Adversarial Training
    """
    generator_adversarial_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    discriminator_adversarial_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    rollout = Rollout(generator, 0.8)

    for epoch_id in range(args.adv_epoch):
        gene_rid_list_data = generator.sample(len(real_train_traj))

        generator_adversarial_dataset_train = GeneratorDataset(gene_rid_list_data)
        generator_adversarial_dataloader_train = DataLoader(generator_adversarial_dataset_train, batch_size=32, shuffle=True, collate_fn=generator_adversarial_collate_fn, num_workers=4, pin_memory=True)

        loss_sum, loss_cnt = 0, 0
        for batch_id, (input_traj, input_time, input_len, label_traj, whole_traj, whole_time, whole_len) in enumerate(tqdm(generator_adversarial_dataloader_train, ncols=100, desc='adversarial training generator')):
            input_traj = input_traj.to(device)
            input_time = input_time.to(device)
            input_len = input_len.to(device)
            label_traj = label_traj.to(device)
            whole_traj = whole_traj.to(device)
            whole_time = whole_time.to(device)
            whole_len = whole_len.to(device)

            logits = generator(input_traj, input_time, input_len)
            rewards = rollout.get_reward(whole_traj, whole_len, args.num_Monte_Carlo, discriminator)

            loss = F.cross_entropy(logits.transpose(1, 2), label_traj, ignore_index=config['num_rid'], reduction='none')

            loss = loss * rewards
            loss = loss.mean()

            loss_sum += loss.item()
            loss_cnt += 1

            loss.backward()
            generator_adversarial_optimizer.step()
            generator_adversarial_optimizer.zero_grad()

            if args.tensorboard:
                writer.add_scalar('generator_adversarial_rewards', rewards.mean().item(), len(generator_adversarial_dataloader_train) * epoch_id + batch_id)
                writer.add_scalar('generator_adversarial_loss', loss.item(), len(generator_adversarial_dataloader_train) * epoch_id + batch_id)

        rollout.update_params()
        torch.save(generator.state_dict(), f'./save/{dataset}/adversarial_trained_generator_epoch{epoch_id}.pth')
        print(f'Adversarial Training Generator, epoch: {epoch_id}, avg_loss: {loss_sum/loss_cnt}')

        fake_rid_list_data = generator.sample(len(real_train_traj))

        discriminator_adversarial_dataset_train = DiscriminatorDataset(real_train_traj['rid_list'].tolist(), fake_rid_list_data)
        discriminator_adversarial_dataloader_train = DataLoader(discriminator_adversarial_dataset_train, batch_size=32, shuffle=True, collate_fn=discriminator_collate_fn, num_workers=4, pin_memory=True)

        loss_sum, loss_cnt = 0, 0
        for batch_id, (input_traj, input_len, label) in enumerate(tqdm(discriminator_adversarial_dataloader_train, ncols=100, desc='adversarial training discriminator')):
            input_traj = input_traj.to(device)
            input_len = input_len.to(device)
            label = label.to(device)

            logits = discriminator(input_traj, input_len)
            loss = F.cross_entropy(logits, label)

            loss_sum += loss.item()
            loss_cnt += 1

            loss.backward()
            discriminator_adversarial_optimizer.step()
            discriminator_adversarial_optimizer.zero_grad()

            if args.tensorboard:
                writer.add_scalar('discriminator_adversarial_loss', loss, len(discriminator_dataloader_train) * epoch_id + batch_id)

        torch.save(discriminator.state_dict(), f'./save/{dataset}/adversarial_trained_discriminator_epoch{epoch_id}.pth')
        print(f'Adversarial Training Discriminator, epoch: {epoch_id}, avg_loss: {loss_sum/loss_cnt}')
