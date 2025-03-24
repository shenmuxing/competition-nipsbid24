# samplers.py

from torch.utils.data import Sampler
import numpy as np
import os
import gc

class RewardBasedSampler(Sampler):
    def __init__(self, dataset, threshold, alpha, beta, memmap_dir='memmap_data', force_recreate=True):
        """
        初始化采样器。
        
        参数:
        - dataset: 数据集对象。
        - threshold: 奖励的阈值，用于区分高奖励和低奖励。
        - alpha: 高奖励样本的比例。
        - beta: 低奖励样本的比例。
        - memmap_dir: 存储 memmap 文件的目录。
        - force_recreate: 是否强制重新创建 memmap 文件。
        """
        self.dataset = dataset
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.memmap_dir = memmap_dir
        self.force_recreate = force_recreate  # 是否强制重新创建 memmap 文件
        os.makedirs(self.memmap_dir, exist_ok=True)
        self.setup()

    def setup(self):
        self.load_or_compute_indices()
        self.reset()

    def load_or_compute_indices(self):
        done_memmap_path = os.path.join(self.memmap_dir, 'done_indices.dat')
        not_done_memmap_path = os.path.join(self.memmap_dir, 'not_done_indices.dat')
        high_reward_memmap_path = os.path.join(self.memmap_dir, 'high_reward_indices.dat')
        low_reward_memmap_path = os.path.join(self.memmap_dir, 'low_reward_indices.dat')

        # 如果 force_recreate 为 True，则删除已有的 memmap 文件
        if self.force_recreate:
            for path in [done_memmap_path, not_done_memmap_path, high_reward_memmap_path, low_reward_memmap_path]:
                if os.path.exists(path):
                    os.remove(path)

        if not os.path.exists(done_memmap_path) or not os.path.exists(not_done_memmap_path):
            all_dones = self.dataset.get_all_dones()
            all_rewards = self.dataset.get_all_rewards()

            self.indices = np.arange(len(all_dones))

            done_indices = self.indices[all_dones == 1]
            not_done_indices = self.indices[all_dones != 1]
            high_reward_indices = not_done_indices[all_rewards[not_done_indices] > self.threshold]
            low_reward_indices = not_done_indices[all_rewards[not_done_indices] <= self.threshold]

            # Save indices to memmap files
            done_memmap = np.memmap(done_memmap_path, dtype='int32', mode='w+', shape=done_indices.shape)
            done_memmap[:] = done_indices
            done_memmap.flush()

            not_done_memmap = np.memmap(not_done_memmap_path, dtype='int32', mode='w+', shape=not_done_indices.shape)
            not_done_memmap[:] = not_done_indices
            not_done_memmap.flush()

            high_reward_memmap = np.memmap(high_reward_memmap_path, dtype='int32', mode='w+', shape=high_reward_indices.shape)
            high_reward_memmap[:] = high_reward_indices
            high_reward_memmap.flush()

            low_reward_memmap = np.memmap(low_reward_memmap_path, dtype='int32', mode='w+', shape=low_reward_indices.shape)
            low_reward_memmap[:] = low_reward_indices
            low_reward_memmap.flush()

            del done_memmap
            del not_done_memmap
            del high_reward_memmap
            del low_reward_memmap
            self.done_indices = np.memmap(done_memmap_path, dtype='int32', mode='r')
            self.not_done_indices = np.memmap(not_done_memmap_path, dtype='int32', mode='r')
            self.high_reward_indices = np.memmap(high_reward_memmap_path, dtype='int32', mode='r')
            self.low_reward_indices = np.memmap(low_reward_memmap_path, dtype='int32', mode='r')

        else:
            print('else1')
            self.done_indices = np.memmap(done_memmap_path, dtype='int32', mode='r')
            self.not_done_indices = np.memmap(not_done_memmap_path, dtype='int32', mode='r')
            self.high_reward_indices = np.memmap(high_reward_memmap_path, dtype='int32', mode='r')
            print('else1:type(high_reward_indices)',type(high_reward_indices))
            self.low_reward_indices = np.memmap(low_reward_memmap_path, dtype='int32', mode='r')
        
    def reset(self):

        # 计算采样数量
        high_reward_sample_size = int(len(self.high_reward_indices) * self.alpha)
        low_reward_sample_size = int(len(self.low_reward_indices) * self.beta)
        print('begin to sample')
        
        # 采样并合并索引
        sampled_indices_path = os.path.join(self.memmap_dir, 'sampled_indices.dat')
        total_sample_size = len(self.done_indices) + high_reward_sample_size + low_reward_sample_size
        print('total_sample_size:', total_sample_size)
        gc.collect() # 释放一点内存
        self.sampled_indices = np.memmap(sampled_indices_path, dtype=np.int32, mode='w+', shape=(total_sample_size,))

        # for done_indices
        current_index = 0  # epoch来这里，32g
        self.sampled_indices[current_index: current_index + len(self.done_indices)] = self.done_indices
        self.sampled_indices.flush()
        # for high_reward_indices
        current_index += len(self.done_indices)

        high_reward_samples = np.random.choice(self.high_reward_indices, high_reward_sample_size, replace=False)
        self.sampled_indices[current_index:current_index + high_reward_sample_size] = high_reward_samples
        self.sampled_indices.flush()
        print('finished high_reward_indices')
        # for low_reward_indices
        current_index += high_reward_sample_size
        low_reward_samples = np.random.choice(self.low_reward_indices, low_reward_sample_size, replace=False)
        self.sampled_indices[current_index:] = low_reward_samples
        self.sampled_indices.flush()
        # self.sampled_indices = np.concatenate([
        #     self.done_indices,
        #     np.random.choice(self.high_reward_indices, high_reward_sample_size, replace=False), # 切分
        #     np.random.choice(self.low_reward_indices, low_reward_sample_size, replace=False)
        # ])   # 40min
        print('finished sample1')
        # 排序索引
        self.sampled_indices.sort()

        # 计算权重
        all_dones = self.dataset.get_all_dones()
        all_rewards = self.dataset.get_all_rewards()
        print('begin to compute weights')
        self.weights = np.ones(len(self.sampled_indices))
        self.weights[all_dones[self.sampled_indices] != 1] = np.where(
            all_rewards[self.sampled_indices[all_dones[self.sampled_indices] != 1]] > self.threshold,
            1 / self.alpha,
            1 / self.beta
        )

    def __iter__(self):
        print('begin to iter samplers')
        return iter(self.sampled_indices)

    def __len__(self):
        return len(self.sampled_indices)