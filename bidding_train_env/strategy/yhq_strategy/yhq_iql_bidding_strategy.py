# yhq_iql_bidding_strategy.py
# 在bidding方法中实现了竞价过程
import numpy as np
import torch
import pickle
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
import os
import json
from utils.zjy_config import get_config
from bidding_train_env.zjy_train.iql.iql import IQL,Q,V,Actor  # 确保这个导入路径是正确的

class IqlBiddingStrategy(BaseBiddingStrategy):
    """
    IQL Strategy
    """

    def __init__(self, budget=100, name="Iql-PlayerStrategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)

        cfg = get_config()

        self.normalize_params = self.load_normalization_params(cfg.normalized_params_path)

        # 创建模型实例
        state_dim = len(self.normalize_params['state_mean'])  # 假设状态维度与归一化参数的长度相同        
        self.model = IQL(num_of_states=state_dim, cfg=cfg)

         # 加载模型状态字典
        state_dict = torch.load(cfg.model_file, map_location='cpu')
        
        # 创建模型实例
        state_dim = len(self.normalize_params['state_mean'])
        self.model = IQL(num_of_states=state_dim, cfg=cfg)
        
        # 确保所有模型参数都在CPU上
        self.model.to('cpu')
        
        # 单独加载每个子模型的状态，并确保它们都在CPU上
        self.model.critic1.load_state_dict({k: v.cpu() for k, v in state_dict['critic1'].items()})
        self.model.critic2.load_state_dict({k: v.cpu() for k, v in state_dict['critic2'].items()})
        self.model.value_net.load_state_dict({k: v.cpu() for k, v in state_dict['value_net'].items()})
        self.model.actors.load_state_dict({k: v.cpu() for k, v in state_dict['actors'].items()})
        
        # 将模型设置为评估模式
        self.model.eval()

    def load_normalization_params(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """
        Bids for all the opportunities in a delivery period

        parameters:
        @timeStepIndex: the index of the current decision time step.
        @pValues: the conversion action probability.
        @pValueSigmas: the prediction probability uncertainty.
        @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
        @historyBid: the advertiser's history bids for each opportunity.
        @historyAuctionResult: the history auction results for each opportunity.
        @historyImpressionResult: the history impression result for each opportunity.
        @historyLeastWinningCosts: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """
        timeStepIndexNum = 48
        timeleft = (timeStepIndexNum - timeStepIndex) / timeStepIndexNum
        bgtleft = self.remaining_budget / self.budget if self.budget > 0 else 0

        # Calculate lags (assuming the most recent data is at the end of the list)
        bid_lags = [np.mean(historyBid[-i]) if i <= len(historyBid) else 0 for i in range(1, 4)]
        leastWinningCost_lags = [np.mean(historyLeastWinningCost[-i]) if i <= len(historyLeastWinningCost) else 0 for i in range(1, 4)]
        pValue_lags = [np.mean(historyPValueInfo[-i][:, 0]) if i <= len(historyPValueInfo) else 0 for i in range(1, 4)]
        conversionAction_lags = [np.mean(historyImpressionResult[-i][:, 1]) if i <= len(historyImpressionResult) else 0 for i in range(1, 4)]
        xi_lags = [np.mean(historyAuctionResult[-i][:, 0]) if i <= len(historyAuctionResult) else 0 for i in range(1, 4)]

        # Calculate cumulative values
        cumulative_realCost = self.budget - self.remaining_budget
        cumulative_conversions = sum(sum(imp[:, 1]) for imp in historyImpressionResult)
        cumulative_pValue = sum(sum(pv[:, 0]) for pv in historyPValueInfo)

        # Current pValue
        current_pValue = np.mean(pValues)

        state = np.array([
            timeleft, bgtleft,
            bid_lags[0], bid_lags[1], bid_lags[2],
            leastWinningCost_lags[0], leastWinningCost_lags[1], leastWinningCost_lags[2],
            current_pValue, pValue_lags[0], pValue_lags[1], pValue_lags[2],
            conversionAction_lags[0], conversionAction_lags[1], conversionAction_lags[2],
            xi_lags[0], xi_lags[1], xi_lags[2],
            cumulative_realCost, cumulative_conversions, cumulative_pValue,
            self.budget, self.cpa, self.category
        ])

        # 标准化
        state = (state - self.normalize_params['state_mean']) / self.normalize_params['state_std']

        # 转换为tensor
        state = torch.tensor(state, dtype=torch.float)
        
        alpha = self.model(state)
        alpha = alpha.cpu().numpy()
        bids = alpha * pValues

        return bids
