# iql_dataloader.py
import psutil
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from itertools import cycle
import gc
import pyarrow.feather as feather

from bidding_train_env.zjy_train.iql.samplers import RewardBasedSampler


class RLDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_path = cfg.training_data_path
        self.memmap_path = cfg.memmap_data_path
        self.file_list = [f for f in os.listdir(self.data_path) if f.endswith(('.csv', '.feather'))]
        self.normalize_params = None
        if cfg.use_normalization:
            self.compute_normalization_params()
        print('finised compute_normalization_params')
        self.current_file = None
        self.current_data = None
        self.done_cache = {}
        self.reward_cache = {}
        self.has_saved_memmap = False

        # Initialize dataset info
        self._initialize_dataset_info()

    def get_feather_row_count(self, file_path):  # 使用pyarrow获取 feather 文件的元数据，快速获取文件大小
        with open(file_path, 'rb') as f:
            reader = feather.read_table(f, columns=[])
        return reader.num_rows

    def _initialize_dataset_info(self):
        memmap_file_sizes = os.path.join(self.memmap_path, 'file_sizes.dat')
        memmap_dones = os.path.join(self.memmap_path, 'all_dones.dat')
        memmap_rewards = os.path.join(self.memmap_path, 'all_rewards.dat')

        if (os.path.exists(memmap_file_sizes) and
            os.path.exists(memmap_dones) and
            os.path.exists(memmap_rewards)):
            print('load memmap files')
            self.file_sizes = np.memmap(memmap_file_sizes, dtype='int32', mode='r')
            print('finished loadfile_sizes')
            self.all_dones = np.memmap(memmap_dones, dtype='bool', mode='r')
            print('finished load all_dones')
            self.all_rewards = np.memmap(memmap_rewards, dtype='float32', mode='r')
            print('finished load all_rewards')
            self.has_saved_memmap = True
        else:
            self.file_sizes = []
            self.all_dones = []
            self.all_rewards = []
            total_size = 0
            # Initialize memmap files
            print('begin to calculate file sizes')
            total_size = sum(self.get_feather_row_count(os.path.join(self.data_path, file)) for file in self.file_list)
            print('finished calculating file sizes')
            file_sizes_memmap = np.memmap(memmap_file_sizes, dtype='int32', mode='w+', shape=(len(self.file_list),))
            all_dones_memmap = np.memmap(memmap_dones, dtype='bool', mode='w+', shape=(total_size,))
            all_rewards_memmap = np.memmap(memmap_rewards, dtype='float32', mode='w+', shape=(total_size,))
            current_index = 0
            for i, file in enumerate(self.file_list):
                print('write_file',file)
                file_path = os.path.join(self.data_path, file)
                df = self.load_file_lazy(file_path)
                file_size = len(df)
                # Update file sizes
                file_sizes_memmap[i] = file_size
                # Update dones and rewards
                all_dones_memmap[current_index:current_index+file_size] = df['done'].values
                all_rewards_memmap[current_index:current_index+file_size] = df['reward_continuous'].values
                current_index += file_size
                # Force writing to disk
                all_dones_memmap.flush()
                all_rewards_memmap.flush()

                gc.collect()
            file_sizes_memmap.flush()
            
        # Load the data into the instance variables
        self.file_sizes = np.memmap(memmap_file_sizes, dtype='int32', mode='r')
        self.all_dones = np.memmap(memmap_dones, dtype='bool', mode='r')
        self.all_rewards = np.memmap(memmap_rewards, dtype='float32', mode='r')

        self.total_size = sum(self.file_sizes)
        self.file_cumsum = np.cumsum([0] + list(self.file_sizes))

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.file_cumsum, idx, side='right') - 1
        row_idx = idx - self.file_cumsum[file_idx]

        if self.current_file != file_idx:
            self.load_file(file_idx)

        row = self.current_data.iloc[row_idx]
        return self.process_row(row)

    def load_file(self, file_idx):
        file_path = os.path.join(self.data_path, self.file_list[file_idx])
        self.current_file = file_idx
        self.current_data = None
        gc.collect()
        self.current_data = self.load_file_lazy(file_path)

    @staticmethod
    def load_file_lazy(file_path):
        print("loading file: ", file_path)
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:  # .feather
            return pd.read_feather(file_path)

    def process_row(self, row):
        state = np.array(eval(row['state']) if isinstance(row['state'], str) else row['state'])
        next_state = np.array(eval(row['next_state']) if isinstance(row['next_state'], str) else row['next_state'])
        if next_state.ndim == 0:  # 说明是None
            next_state = state        
        
        if self.cfg.use_normalization:
            state = (state - self.normalize_params['state_mean']) / self.normalize_params['state_std']
            next_state = (next_state - self.normalize_params['state_mean']) / self.normalize_params['state_std']

        return {
            'state': torch.FloatTensor(state),
            'action': torch.FloatTensor([row['action']]),
            'reward': torch.FloatTensor([200 * row['reward_continuous']]),
            'next_state': torch.FloatTensor(next_state),
            'done': torch.FloatTensor([row['done']])
        }

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 /1024  # 返回以GB为单位的内存使用量

    def compute_normalization_params(self):
        first_file = self.file_list[0]
        file_path = os.path.join(self.data_path, first_file)
        data = feather.read_feather(file_path, columns=['state']) #只读一列，self.load_file_lazy(file_path)
        try:
            states = np.stack(data['state'].values)
        except:
            print('str is in data, try eval')
            states = np.stack(data['state'].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x).values)
        self.normalize_params = {
            'state_mean': np.mean(states, axis=0),
            'state_std': np.std(states, axis=0)
        }
        self.save_normalization_params()

    def save_normalization_params(self):
        print("Normalize params content:", self.normalize_params)
        if not os.path.exists(os.path.dirname(self.cfg.normalized_params_path)):
            os.makedirs(os.path.dirname(self.cfg.normalized_params_path))
        with open(self.cfg.normalized_params_path, 'wb') as f:
            pickle.dump(self.normalize_params, f)

    @staticmethod
    def load_normalization_params(cfg):
        with open(cfg.normalized_params_path, 'rb') as f:
            return pickle.load(f)

    def get_file_sizes(self):
        return self.file_sizes

    def get_all_dones(self):
        return self.all_dones

    def get_all_rewards(self):
        return self.all_rewards
    
    def get_data_info(self):
        return len(self.file_list), self.total_size
    
def collate_fn(batch):
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }

class RLDataLoader(DataLoader):
    def __init__(self, cfg):
        self.dataset = RLDataset(cfg)
        if cfg.use_reward_based_sampler:
            print('begin to sample by reward')
            self.sampler = RewardBasedSampler(self.dataset, cfg.reward_threshold, cfg.alpha, cfg.beta)
            print('finished sampling by reward')
        else:
            self.sampler = None  # 使用默认的随机采样
        super().__init__(
            self.dataset,
            batch_size=cfg.batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 0
        )

    def __iter__(self):
        if self.sampler:
            self.sampler.reset()  # 在每个 epoch 开始前重新采样
        return super().__iter__()

    def get_state_dim(self):
        return self.dataset[0]['state'].shape[0]