# run.zjy.zjy_run_attentioniql2.py
import numpy as np
import logging
import sys
import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

from bidding_train_env.zjy_train.attention_iql2.replay_buffer import ReplayBuffer
from bidding_train_env.zjy_train.attention_iql2.aiql import AIQL

from torch.utils.data import DataLoader, Subset
from copy import deepcopy
# 设置打印选项
np.set_printoptions(suppress=True, precision=4)

class EarlyStopping:
    """
    实现早停机制
    """
    def __init__(self, patience=5, verbose=False, delta=0.0):
        """
        Args:
            patience (int): 多少个epoch没有提升就停止训练
            verbose (bool): 是否输出日志
            delta (float): 提升的最小变化量
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, loss):
        score = -loss  # 假设损失越小越好，取负数作为score

        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                logging.getLogger(__name__).info(f'Initial loss: {loss:.4f}')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.getLogger(__name__).info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

            if self.verbose:
                logging.getLogger(__name__).info(f'Loss improved to: {loss:.4f}')

def safe_literal_eval(val):
    """
    安全地将字符串转换为Python对象
    """
    if pd.isna(val):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        logging.getLogger(__name__).warning(f'Error parsing value: {val}')
        return val

def preprocess_data(df, is_docker):
    """
    预处理数据，包括转换状态信息为Tensor格式
    """
    if is_docker:
        df["state_current_informations"] = df["state_current_informations"].apply(safe_literal_eval)
        df["state_history_informations"] = df["state_history_informations"].apply(safe_literal_eval)
        df["next_state_current_information"] = df["next_state_current_information"].apply(safe_literal_eval)
        df["next_state_history_information"] = df["next_state_history_information"].apply(safe_literal_eval)

    # 转换当前信息为Tensor
    def to_tensor_current(row, D_curr):
        state = row['state_current_informations']
        if len(state) == 0:
            return torch.zeros((1, 1, D_curr))  # 根据D_curr设定
        return torch.tensor(state, dtype=torch.float).unsqueeze(0).unsqueeze(0)  # [1,1,D_curr]

    # 转换历史信息为Tensor
    def to_tensor_history(row, D_hist):
        history = row['state_history_informations']
        if len(history) == 0 or len(history[0]) == 0:
            return torch.zeros((1, 1, D_hist))  # 根据D_hist设定
        tensor_history = torch.tensor(history.tolist(), dtype=torch.float).reshape(-1, 1, D_hist)  # [T,1,D_hist]
        return tensor_history

    # 确保DataFrame不为空
    if len(df) > 0:
        D_curr = len(df.loc[0, 'state_current_informations'])
        D_hist = len(df.loc[0, 'state_history_informations'])  # 修改：获取历史信息的单个元素长度
    else:
        # 如果DataFrame为空，设置默认值或抛出异常
        raise ValueError("DataFrame is empty")

    # 修改：将D_curr和D_hist作为参数传递给apply方法
    df['current_tensor'] = df.apply(lambda row: to_tensor_current(row, D_curr), axis=1)
    df['history_tensor'] = df.apply(lambda row: to_tensor_history(row, D_hist), axis=1)

    return df

def train_iql_model(cfg, logger, train_replay_buffer, test_replay_buffer):
    """
    训练IQL模型
    """
    model = AIQL(d_curr= cfg.D_curr, d_hist = cfg.D_hist, d_model = cfg.attention_d_model, cfg=cfg)
    model = model.to(model.device)

    early_stopping = EarlyStopping(patience=cfg.iql_patience, verbose=True)

    num_epochs = cfg.iql_num_epochs
    batch_size = cfg.attention_batch_size

    # 构建训练数据集和测试数据集
    train_dataset = list(train_replay_buffer.memory)
    test_dataset = list(test_replay_buffer.memory)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = cfg.num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        q_total_loss = 0.0
        v_total_loss = 0.0
        a_total_loss = 0.0
        for state_current_informations,state_history_informations, action, reward, next_state_current_informations,\
              next_state_history_informations, done, state_current_informations_other,next_state_current_informations_other \
            in tqdm(train_dataloader, desc=f'Critic Training Epoch {epoch+1}/{num_epochs}',mininterval=1):
            state_current_informations = torch.tensor(state_current_informations, dtype=torch.float).to(model.device).squeeze(0)
            state_history_informations = torch.tensor(state_history_informations, dtype=torch.float).to(model.device).squeeze(0)
            action = (torch.tensor(action, dtype=torch.float).to(model.device)-80)/50
            reward = torch.tensor(reward, dtype=torch.float).to(model.device)
            next_state_current_informations = torch.tensor(next_state_current_informations, dtype=torch.float).to(model.device).squeeze(0)
            next_state_history_informations = torch.tensor(next_state_history_informations, dtype=torch.float).to(model.device).squeeze(0)
            done = torch.tensor(done, dtype=torch.float).to(model.device)
            state_current_informations_other = torch.tensor(state_current_informations_other, dtype=torch.float).to(model.device).squeeze(0)
            next_state_current_informations_other = torch.tensor(next_state_current_informations_other, dtype=torch.float).to(model.device).squeeze(0)

            q_loss, v_loss, a_loss = model.step(state_current_informations, state_history_informations, action, reward, next_state_current_informations, 
                                                next_state_history_informations, done,state_current_informations_other = state_current_informations_other, 
                                                next_state_current_informations_other = next_state_current_informations_other, mode = 'critic')

            loss = q_loss + v_loss
            q_total_loss += q_loss.item()
            v_total_loss += v_loss.item()
            # a_total_loss += a_loss.item()
            epoch_loss += loss.item() 

        epoch_loss /= len(train_dataloader.dataset)
        logger.info(f'IQL Train Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
        print(f'Train Q loss: {q_total_loss/len(train_dataloader):.4f}, V loss: {v_total_loss/len(train_dataloader):.4f}')

        # 在测试集上评估
        model.eval()
        test_loss = 0.0
        test_q_loss = 0.0
        test_v_loss = 0.0
        test_a_loss = 0.0
        with torch.no_grad():
            for state_current_informations,state_history_informations, action, reward, next_state_current_informations, \
                next_state_history_informations, done,state_current_informations_other,next_state_current_informations_other \
                in test_dataloader:
                state_current_informations = torch.tensor(state_current_informations, dtype=torch.float).to(model.device).squeeze(0)
                state_history_informations = torch.tensor(state_history_informations, dtype=torch.float).to(model.device).squeeze(0)
                action = (torch.tensor(action, dtype=torch.float).to(model.device)-80)/50
                reward = torch.tensor(reward, dtype=torch.float).to(model.device)
                next_state_current_informations = torch.tensor(next_state_current_informations, dtype=torch.float).to(model.device).squeeze(0)
                next_state_history_informations = torch.tensor(next_state_history_informations, dtype=torch.float).to(model.device).squeeze(0)
                done = torch.tensor(done, dtype=torch.float).to(model.device)
                state_current_informations_other = torch.tensor(state_current_informations_other, dtype=torch.float).to(model.device).squeeze(0)
                next_state_current_informations_other = torch.tensor(next_state_current_informations_other, dtype=torch.float).to(model.device).squeeze(0)

                q_loss, v_loss, a_loss = model.step(state_current_informations, state_history_informations, action, reward, next_state_current_informations, 
                                                    next_state_history_informations, done,state_current_informations_other = state_current_informations_other, 
                                                        next_state_current_informations_other = next_state_current_informations_other,mode = 'critic')
                loss = q_loss + v_loss 
                test_q_loss += q_loss.item()
                test_v_loss += v_loss.item()
                # test_a_loss += a_loss.item()
                test_loss += loss.item()

        test_loss /= len(test_dataloader.dataset)
        logger.info(f'IQL Test Epoch {epoch+1}, Loss: {test_loss:.4f}')
        print(f'Test Q loss: {test_q_loss/len(test_dataloader):.4f}, V loss: {test_v_loss/len(test_dataloader):.4f}')

        early_stopping(test_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered for IQL model.")
            break

    # 使用测试集数据进行额外训练
    logger.info(f"Training on test set for {cfg.test_set_train_epoches} epochs")


    for epoch in range(cfg.test_set_train_epoches):
        model.train()
        epoch_loss = 0.0
        for state_current_informations,state_history_informations, action, reward, next_state_current_informations, \
            next_state_history_informations, done,state_current_informations_other,next_state_current_informations_other \
            in tqdm(test_dataloader, desc=f'Critic Test Set Training Epoch {epoch+1}/{cfg.test_set_train_epoches}',mininterval=1):
            state_current_informations = torch.tensor(state_current_informations, dtype=torch.float).to(model.device).squeeze(0)
            state_history_informations = torch.tensor(state_history_informations, dtype=torch.float).to(model.device).squeeze(0)
            action = (torch.tensor(action, dtype=torch.float).to(model.device)-80)/50
            reward = torch.tensor(reward, dtype=torch.float).to(model.device)
            next_state_current_informations = torch.tensor(next_state_current_informations, dtype=torch.float).to(model.device).squeeze(0)
            next_state_history_informations = torch.tensor(next_state_history_informations, dtype=torch.float).to(model.device).squeeze(0)
            done = torch.tensor(done, dtype=torch.float).to(model.device)
            state_current_informations_other = torch.tensor(state_current_informations_other, dtype=torch.float).to(model.device).squeeze(0)
            next_state_current_informations_other = torch.tensor(next_state_current_informations_other, dtype=torch.float).to(model.device).squeeze(0)

            q_loss, v_loss, a_loss = model.step(state_current_informations, state_history_informations, action, reward, next_state_current_informations, 
                                                next_state_history_informations, done,state_current_informations_other = state_current_informations_other, 
                                                    next_state_current_informations_other = next_state_current_informations_other,mode='critic')
            loss = q_loss + v_loss
            epoch_loss += loss.item()

        epoch_loss /= len(test_dataloader.dataset)
        logger.info(f'IQL Test Set Training Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

    # 构建训练数据集和测试数据集
    train_dataset = list(train_replay_buffer.memory)
    test_dataset = list(test_replay_buffer.memory)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers = cfg.num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers = cfg.num_workers)

    early_stopping = EarlyStopping(patience=cfg.iql_patience, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        q_total_loss = 0.0
        v_total_loss = 0.0
        a_total_loss = 0.0
        for state_current_informations,state_history_informations, action, reward, next_state_current_informations,\
              next_state_history_informations, done, state_current_informations_other,next_state_current_informations_other \
            in tqdm(train_dataloader, desc=f'Actor Training Epoch {epoch+1}/{num_epochs}',mininterval=1):
            state_current_informations = torch.tensor(state_current_informations, dtype=torch.float).to(model.device).squeeze(0)
            state_history_informations = torch.tensor(state_history_informations, dtype=torch.float).to(model.device).squeeze(0)
            action = (torch.tensor(action, dtype=torch.float).to(model.device)-80)/50
            reward = torch.tensor(reward, dtype=torch.float).to(model.device)
            next_state_current_informations = torch.tensor(next_state_current_informations, dtype=torch.float).to(model.device).squeeze(0)
            next_state_history_informations = torch.tensor(next_state_history_informations, dtype=torch.float).to(model.device).squeeze(0)
            done = torch.tensor(done, dtype=torch.float).to(model.device)
            state_current_informations_other = torch.tensor(state_current_informations_other, dtype=torch.float).to(model.device).squeeze(0)
            next_state_current_informations_other = torch.tensor(next_state_current_informations_other, dtype=torch.float).to(model.device).squeeze(0)

            q_loss, v_loss, a_loss = model.step(state_current_informations, state_history_informations, action, reward, next_state_current_informations, 
                                                next_state_history_informations, done,state_current_informations_other = state_current_informations_other, 
                                                next_state_current_informations_other = next_state_current_informations_other, mode = 'actor')
            # if a_loss > 100:
            #     print(a_loss)
            loss = a_loss
            q_total_loss += q_loss.item()
            v_total_loss += v_loss.item()
            a_total_loss += a_loss.item()
            epoch_loss += loss.item()

        epoch_loss /= len(train_dataloader.dataset)
        logger.info(f'IQL Train Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
        print(f'Train Q loss: {q_total_loss/len(train_dataloader):.4f}, V loss: {v_total_loss/len(train_dataloader):.4f}, A loss: {a_total_loss/len(train_dataloader):.4f}')

        # 在测试集上评估
        model.eval()
        test_loss = 0.0
        test_q_loss = 0.0
        test_v_loss = 0.0
        test_a_loss = 0.0
        with torch.no_grad():
            for state_current_informations,state_history_informations, action, reward, next_state_current_informations, \
                next_state_history_informations, done,state_current_informations_other,next_state_current_informations_other \
                in test_dataloader:
                state_current_informations = torch.tensor(state_current_informations, dtype=torch.float).to(model.device).squeeze(0)
                state_history_informations = torch.tensor(state_history_informations, dtype=torch.float).to(model.device).squeeze(0)
                action = (torch.tensor(action, dtype=torch.float).to(model.device)-80)/50
                reward = torch.tensor(reward, dtype=torch.float).to(model.device)
                next_state_current_informations = torch.tensor(next_state_current_informations, dtype=torch.float).to(model.device).squeeze(0)
                next_state_history_informations = torch.tensor(next_state_history_informations, dtype=torch.float).to(model.device).squeeze(0)
                done = torch.tensor(done, dtype=torch.float).to(model.device)
                state_current_informations_other = torch.tensor(state_current_informations_other, dtype=torch.float).to(model.device).squeeze(0)
                next_state_current_informations_other = torch.tensor(next_state_current_informations_other, dtype=torch.float).to(model.device).squeeze(0)

                q_loss, v_loss, a_loss = model.step(state_current_informations, state_history_informations, action, reward, next_state_current_informations, 
                                                    next_state_history_informations, done,state_current_informations_other = state_current_informations_other, 
                                                        next_state_current_informations_other = next_state_current_informations_other,mode = 'actor')
                loss = a_loss
                test_q_loss += q_loss.item()
                test_v_loss += v_loss.item()
                test_a_loss += a_loss.item()
                test_loss += loss.item()

        test_loss /= len(test_dataloader.dataset)
        logger.info(f'IQL Test Epoch {epoch+1}, Loss: {test_loss:.4f}')
        print(f'Test Q loss: {test_q_loss/len(test_dataloader):.4f}, V loss: {test_v_loss/len(test_dataloader):.4f}, A loss: {test_a_loss/len(test_dataloader):.4f}')

        early_stopping(test_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered for IQL model.")
            break

    # 使用测试集数据进行额外训练
    logger.info(f"Training on test set for {cfg.test_set_train_epoches} epochs")
    for epoch in range(cfg.test_set_train_epoches):
        model.train()
        
        epoch_loss = 0.0
        for state_current_informations,state_history_informations, action, reward, next_state_current_informations, \
            next_state_history_informations, done,state_current_informations_other,next_state_current_informations_other \
            in tqdm(test_dataloader, desc=f'Actor Test Set Training Epoch {epoch+1}/{cfg.test_set_train_epoches}',mininterval=1):
            state_current_informations = torch.tensor(state_current_informations, dtype=torch.float).to(model.device).squeeze(0)
            state_history_informations = torch.tensor(state_history_informations, dtype=torch.float).to(model.device).squeeze(0)
            action = (torch.tensor(action, dtype=torch.float).to(model.device)-80)/50
            reward = torch.tensor(reward, dtype=torch.float).to(model.device)
            next_state_current_informations = torch.tensor(next_state_current_informations, dtype=torch.float).to(model.device).squeeze(0)
            next_state_history_informations = torch.tensor(next_state_history_informations, dtype=torch.float).to(model.device).squeeze(0)
            done = torch.tensor(done, dtype=torch.float).to(model.device)
            state_current_informations_other = torch.tensor(state_current_informations_other, dtype=torch.float).to(model.device).squeeze(0)
            next_state_current_informations_other = torch.tensor(next_state_current_informations_other, dtype=torch.float).to(model.device).squeeze(0)

            q_loss, v_loss, a_loss = model.step(state_current_informations, state_history_informations, action, reward, next_state_current_informations, 
                                                next_state_history_informations, done,state_current_informations_other = state_current_informations_other, 
                                                    next_state_current_informations_other = next_state_current_informations_other,mode='actor')
            loss =  a_loss
            epoch_loss += loss.item()

        epoch_loss /= len(test_dataloader.dataset)
        logger.info(f'IQL Test Set Training Epoch {epoch+1}, Loss: {epoch_loss:.4f}')


    if model.writer:
        model.close_writer()
    # 保存训练好的IQL模型
    iql_save_path = Path(cfg.iql_save_path)
    iql_save_path.mkdir(parents=True, exist_ok=True)
    model.save_net(iql_save_path)
    logger.info(f'IQL model saved to {iql_save_path}')

    return model

def run_iql(cfg, logger):
    """
    运行IQL模型训练和评估。
    """
    # 1. 加载数据
    if cfg.IS_DOCKER:
        train_data_path = "./data/traffic/zjy_training_data_rlData_folder/training_data_all-rlData.csv"
        df = pd.read_csv(train_data_path)
    else:
        train_data_path = "./data/traffic/zjy_training_data_rlData_folder/training_data_all-rlData.feather"
        df = pd.read_feather(train_data_path)

    logger.info(f'Data loaded from {train_data_path}')
    D_curr = cfg.D_curr
    D_hist = cfg.D_hist
    D_other = cfg.D_other
    use_cuda = cfg.use_cuda
    device = torch.device("cuda:1" if use_cuda and torch.cuda.is_available() else "cpu")
    # 转换当前信息为np.ndarray
    def to_ndarray_current(state, D_curr):
        # if state is None or len(state) == 0:
        if len(state) == 0:
            return np.zeros([1, 1, D_curr])  # 根据D_curr设定
        return np.array(state).reshape(1, 1, D_curr)  # [1,1,D_curr]

    # 转换历史信息为Tensor
    def to_ndarray_history(history, D_hist):
        if history is None or len(history) == 0 or len(history[0]) == 0:
            return np.zeros((1, 1, D_hist))  # 根据D_hist设定
        tensor_history = torch.tensor(history.tolist(), dtype=torch.float).reshape(-1, 1, D_hist).numpy()  # [T,1,D_hist]
        return tensor_history

    # 7. 构建Replay Buffer并划分训练集和测试集
    train_replay_buffer = ReplayBuffer()
    test_replay_buffer = ReplayBuffer()
    
    train_df, test_df = train_test_split(df, train_size=cfg.attention_train_test_split, random_state=42)

    for df, buffer in [(train_df, train_replay_buffer), (test_df, test_replay_buffer)]:
        for index, row in df.iterrows():
            # if len(row['state_current_informations']) != 0:
            state_current_informations = row['state_current_informations']
            state_history_informations = row['state_history_informations']
            state_current_informations_other = row['state_current_others']
            action = row['action']
            reward = row['reward_continuous']
            next_state_current_informations = row['next_state_current_information']
            next_state_history_informations = row['next_state_history_information']
            next_state_current_informations_other = row['next_state_current_others']
            # done = row['done']
            done = row['is_last_step']
            if done != 1:
                buffer.push(to_ndarray_current(state_current_informations, D_curr),to_ndarray_history(state_history_informations, D_hist), np.array([action]),np.array([reward]), 
                            to_ndarray_current(next_state_current_informations, D_curr), to_ndarray_history(next_state_history_informations, D_hist), np.array([done]),
                            to_ndarray_current(state_current_informations_other, D_other), to_ndarray_current(next_state_current_informations_other, D_other),device=device)
            else:
                buffer.push(to_ndarray_current(state_current_informations, D_curr), to_ndarray_history(state_history_informations, D_hist), np.array([action]), 
                            np.array([reward]), to_ndarray_current(state_current_informations, D_curr), to_ndarray_history(state_history_informations, D_hist), np.array([done]),
                            to_ndarray_current(state_current_informations_other, D_other), to_ndarray_current(state_current_informations_other, D_other),device=device)

    logger.info(f'Train Replay Buffer initialized with {len(train_replay_buffer.memory)} transitions.')
    logger.info(f'Test Replay Buffer initialized with {len(test_replay_buffer.memory)} transitions.')

    # 8. 训练IQL模型
    iql_model = train_iql_model(cfg, logger, train_replay_buffer, test_replay_buffer)

    logger.info("IQL training completed successfully.")