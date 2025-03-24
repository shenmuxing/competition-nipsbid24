# zjy_run_iql.py
import numpy as np
import torch
from bidding_train_env.zjy_train.iql.iql_dataloader import RLDataLoader
from bidding_train_env.zjy_train.iql.iql import IQL
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
import gc

def train_iql_model(cfg, logger):
    """
    Train the IQL model using the new dataloader.
    """
    # Initialize dataloader
    print("start to load data")
    dataloader = RLDataLoader(cfg)
    print("finish loading data")
    # Get state dimension
    STATE_DIM = dataloader.get_state_dim()
    
    # Initialize model
    model = IQL(num_of_states=STATE_DIM, cfg=cfg)
    
    # Train model
    train_model_epochs(model, dataloader, cfg, logger)
    
    # Test trained model
    test_trained_model(model, dataloader, logger)

class EarlyStopping:
    def __init__(self, patience=2):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, q_loss, v_loss, a_loss):
        score = -(q_loss + v_loss + a_loss)  # 我们希望损失值越小越好
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train_model_epochs(model, dataloader, cfg, logger):
    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience)
    
    for epoch in range(cfg.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{cfg.num_epochs}")
        epoch_q_loss, epoch_v_loss, epoch_a_loss = 0, 0, 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 1000 == 0:
                gc.collect() # 释放一点内存
            states = batch['state']
            actions = batch['action']
            rewards = batch['reward']
            next_states = batch['next_state']
            dones = batch['done']
            if cfg.use_reward_based_sampler:
                weights = torch.tensor(dataloader.sampler.weights[batch_idx * cfg.batch_size:
                                                                  (batch_idx + 1) * cfg.batch_size]).to(model.device)
            else:
                weights = torch.ones(len(states)).to(model.device)
            
            q_loss, v_loss, a_loss = model.step(states, actions, rewards, next_states, dones, weights)

            epoch_q_loss += q_loss
            epoch_v_loss += v_loss
            epoch_a_loss += a_loss
            num_batches += 1
            
            if (batch_idx + 1) % cfg.log_interval == 0:
                avg_q_loss = epoch_q_loss / num_batches
                avg_v_loss = epoch_v_loss / num_batches
                avg_a_loss = epoch_a_loss / num_batches
                logger.info(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, '
                            f'Avg Q_loss: {avg_q_loss:.4f}, Avg V_loss: {avg_v_loss:.4f}, Avg A_loss: {avg_a_loss:.4f}')

            if cfg.use_profiler and (batch_idx + 1) >= cfg.profile_batch_limit:
                break
            # break

        # 计算平均损失
        avg_q_loss = epoch_q_loss / num_batches
        avg_v_loss = epoch_v_loss / num_batches
        avg_a_loss = epoch_a_loss / num_batches
        
        # 检查是否应该早停
        early_stopping(avg_q_loss, avg_v_loss, avg_a_loss)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
        # 定期保存模型
        if (epoch + 1) % cfg.save_interval == 0:
            save_path = f"{cfg.model_save_path}_epoch_{epoch + 1}"
            model.save_net(save_path)
            logger.info(f"Model saved at epoch {epoch + 1}")
    
    # 训练结束后保存最终模型
    model.save_net(cfg.model_save_path)
    logger.info("Final model saved")
            
def test_trained_model(model, dataloader, logger):
    batch = next(iter(dataloader))
    states = batch['state']
    actions = batch['action']
    rewards = batch['reward']
    next_states = batch['next_state']
    dones = batch['done']
    pred_actions = model.take_actions(states)
    actions = actions.cpu().detach().numpy()
    comparison = np.concatenate((actions, pred_actions), axis=1)
    logger.info("action VS pred action:")
    logger.info(comparison)

def run_iql(cfg, logger):
    """
    Run IQL model training and evaluation.
    """
    train_iql_model(cfg, logger)