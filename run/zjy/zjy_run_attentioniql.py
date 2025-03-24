# run.zjy.zjy_run_attentioniql.py
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

from bidding_train_env.zjy_train.attention_iql.replay_buffer import ReplayBuffer
from bidding_train_env.zjy_train.attention_iql.iql import IQL
from bidding_train_env.zjy_train.attention_iql.attention import AttentionIQL
from torch.utils.data import DataLoader, Subset

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

def generate_pca_target(df):
    """
    使用PCA从`max_action_without_me`和`second_action_without_me`生成一维的target
    """
    pca = PCA(n_components=1)
    actions = df[['max_action_without_me', 'second_action_without_me']].dropna()
    pca.fit(actions)
    # df['target'] = pca.transform(df[['max_action_without_me', 'second_action_without_me']])
    # print(f'PCA explained_variance_ratio: {pca.explained_variance_ratio_}')
    
    df['target'] = df['second_action_without_me'].copy()/200
    print(f"mean of target: {df['target'].mean()}")
    print(f"std of target: {df['target'].std()}")
    return df, pca

def train_attention_model(cfg, df, logger):
    """
    训练Attention模型并返回训练好的模型，同时将数据集按顺序划分为训练集和验证集。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_hist = len(df.loc[0, 'state_history_informations'])
    D_curr = len(df.loc[0, 'state_current_informations'])

    attention_model = AttentionIQL(
        d_hist=D_hist,  # 根据您的数据设定
        d_curr=D_curr,  # 根据您的数据设定
        d_model=cfg.attention_d_model,
        n_heads=cfg.attention_n_heads,
        dropout=cfg.attention_dropout
    ).to(device)

    optimizer = optim.Adam(attention_model.parameters(), lr=cfg.attention_lr)
    criterion = nn.MSELoss()

    early_stopping = EarlyStopping(patience=cfg.attention_patience, verbose=True)

    num_epochs = cfg.attention_num_epochs
    batch_size = cfg.attention_batch_size
    train_test_split_ratio = cfg.attention_train_test_split  # 训练集比例

    # 准备数据集
    dataset = list(zip(df['history_tensor'], df['current_tensor'], df['target']))

    # 计算训练集和验证集的大小
    train_size = int(train_test_split_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # 按顺序划分数据集
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, len(dataset)))

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        # 训练阶段
        attention_model.train()
        epoch_train_loss = 0.0
        for history, current, target in tqdm(train_dataloader, desc=f'Attention Training Epoch {epoch+1}/{num_epochs}'):
            history = history.squeeze(0).to(device)  # [T,1,D_hist]
            current = current.squeeze(0).to(device)  # [1,1, D_curr]
            
            target = target.to(device).squeeze().float()  # [batch_size]

            optimizer.zero_grad()
            output = attention_model(history, current, time_step_index=0)  # 调整time_step_index

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_dataloader)
        logger.info(f'Attention Epoch {epoch+1}, Training Loss: {epoch_train_loss:.4f}')

        # 验证阶段
        attention_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for history, current, target in val_dataloader:
                history = history.squeeze(0).to(device)
                current = current.squeeze(0).to(device)
                target = target.to(device).squeeze().float()

                output = attention_model(history, current, time_step_index=0)

                loss = criterion(output, target)
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_dataloader)
        logger.info(f'Attention Epoch {epoch+1}, Validation Loss: {epoch_val_loss:.4f}')

        # 通过验证损失进行早停
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered for Attention model.")
            break

    # 保存训练好的Attention模型
    attention_save_path = Path(cfg.attention_save_path)
    attention_save_path.mkdir(parents=True, exist_ok=True)
    attention_model.save_net(attention_save_path)
    logger.info(f'Attention model saved to {attention_save_path}')

    return attention_model

def augment_data_with_attention(attention_model, df, cfg):
    """
    使用训练好的Attention模型生成注意力向量，并将其添加到状态中
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention_model.eval()

    augmented_current_states = []
    augmented_next_states = []
    D_curr = len(df.loc[0,'state_current_informations'])
    D_hist = len(df.loc[0,'state_history_informations'])
    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Augmenting data with Attention'):
            # 生成当前状态的注意力向量
            history = row['history_tensor'].to(device).reshape(-1,1,D_hist)  # [T,1,D_hist]
            current = row['current_tensor'].to(device).reshape(1,1,D_curr)  # [1,1, D_curr]
            time_step_index = int(row['timeStepIndex']) if 'timeStepIndex' in row else 0
            current_attention = attention_model.get_attention_vector(history, current, time_step_index=time_step_index).detach().squeeze().cpu().numpy()

            # 生成下一个状态的注意力向量
            next_history = row['next_state_history_information']
            next_current = row['next_state_current_information']
            if (next_history is not None) and (next_current is not None) and  len(next_history) > 0 and len(next_current) > 0:
                next_history = next_history.tolist()
                next_current = next_current.tolist()
                next_history_tensor = torch.tensor(next_history, dtype=torch.float).reshape(-1, 1, D_hist).to(device)
                next_current_tensor = torch.tensor(next_current, dtype=torch.float).reshape(1, 1, D_curr).to(device)
                next_attention = attention_model.get_attention_vector(next_history_tensor, next_current_tensor, time_step_index=time_step_index).detach().squeeze().cpu().numpy()

                # 创建增强后的状态
                augmented_next = np.concatenate([row['next_state_current_information'], next_attention])

            else:
                next_attention = np.zeros(cfg.attention_d_model)  # 填充0
                # augmented_current = None
                augmented_next = None

            augmented_current = np.concatenate([row['state_current_informations'], current_attention])


            augmented_current_states.append(augmented_current)
            augmented_next_states.append(augmented_next)

    df['augmented_current_state'] = augmented_current_states
    df['augmented_next_state'] = augmented_next_states

    return df

def train_iql_model(cfg, logger, replay_buffer):
    """
    训练IQL模型
    """
    states_0 = replay_buffer.memory[0][0]
    model = IQL(num_of_states= states_0.shape[-1] , cfg=cfg)
    model = model.to(model.device)

    early_stopping = EarlyStopping(patience=cfg.iql_patience, verbose=True)

    num_epochs = cfg.iql_num_epochs
    batch_size = cfg.iql_batch_size

    # 构建数据集
    dataset = list(replay_buffer.memory)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        q_total_loss = 0.0
        v_total_loss = 0.0
        a_total_loss = 0.0
        for state, action, reward, next_state, done in tqdm(dataloader, desc=f'IQL Training Epoch {epoch+1}/{num_epochs}'):
            state = torch.tensor(state, dtype=torch.float).to(model.device)
            action = torch.tensor(action, dtype=torch.float).to(model.device)/50
            reward = torch.tensor(reward, dtype=torch.float).to(model.device)
            next_state = torch.tensor(next_state, dtype=torch.float).to(model.device)
            done = torch.tensor(done, dtype=torch.float).to(model.device)

            q_loss, v_loss, a_loss = model.step(state, action, reward, next_state, done, weights=torch.ones_like(reward).to(model.device))
            loss = q_loss + v_loss + a_loss
            q_total_loss += q_loss.item()
            v_total_loss += v_loss.item()
            a_total_loss += a_loss.item()
            epoch_loss += loss.item() * state.size(0)

        epoch_loss /= len(dataloader.dataset)
        logger.info(f'IQL Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
        print(f'Q loss: {q_total_loss/len(dataloader):.4f}, V loss: {v_total_loss/len(dataloader):.4f}, A loss: {a_total_loss/len(dataloader):.4f}')

        early_stopping(epoch_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered for IQL model.")
            break

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

    # 2. 预处理数据
    df = preprocess_data(df, cfg.IS_DOCKER)
    logger.info('Data preprocessed.')

    # 3. 归一化处理 这段应该在Attention model相关的部分进行实现。

    # 4. 生成PCA target
    df, pca = generate_pca_target(df)
    logger.info('PCA target generated.')

    # 5. 训练Attention模型
    attention_model = train_attention_model(cfg, df, logger)

    # 6. 增强数据
    df = augment_data_with_attention(attention_model, df, cfg)
    logger.info('Data augmented with attention vectors.')

    # 7. 构建Replay Buffer
    replay_buffer = ReplayBuffer()
    for index, row in df.iterrows():
        state = row['augmented_current_state']
        action = row['action']
        reward = row['reward']
        next_state = row['augmented_next_state']
        done = row['done']

        if done != 1:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.array(next_state), np.array([done]))
        else:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.zeros_like(state), np.array([done]))

    logger.info(f'Replay Buffer initialized with {len(replay_buffer.memory)} transitions.')

    # 8. 训练IQL模型
    iql_model = train_iql_model(cfg, logger, replay_buffer)

    # 9. 测试IQL模型(略)

    logger.info("IQL training completed successfully.")

