# yhq_iql_bidding_strategy.py
# 在bidding方法中实现了竞价过程
import numpy as np
import torch
import pickle
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
import os
import json
from utils.zjy_config import get_config
from bidding_train_env.zjy_train.attention_iql.iql import IQL,Q,V,Actor  # 确保这个导入路径是正确的
from bidding_train_env.zjy_train.attention_iql.attention import TimeEncoding,AttentionIQL
class IqlBiddingStrategy(BaseBiddingStrategy):
    """
    IQL Strategy
    """

    def __init__(self, budget=100, name="Iql-PlayerStrategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)

        cfg = get_config()

        # self.normalize_params = self.load_normalization_params(cfg.normalized_params_path)

        # 创建模型实例
        # state_dim = len(self.normalize_params['state_mean'])  # 假设状态维度与归一化参数的长度相同     
        self.attention = AttentionIQL(
            d_hist = cfg.D_hist,
            d_curr = cfg.D_curr,
            d_model = cfg.attention_d_model,
            n_heads = cfg.attention_n_heads,
            dropout = cfg.attention_dropout
        )   

        self.model = IQL(num_of_states= cfg.D_curr+cfg.attention_d_model, cfg=cfg)

        self.model.load_net(load_path = cfg.iql_save_path)
        self.model.to('cpu')

        self.attention.load_net(load_path = cfg.attention_save_path)
        self.attention.to('cpu')
        # 将模型设置为评估模式
        self.model.eval()
        self.attention.eval()
        
        self.history_budgets = []
        self.historical_adslot1_rates = []
        self.historical_adslot2_rates = []
        self.historical_adslot3_rates = []

    def load_normalization_params(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def reset(self):

        self.remaining_budget = self.budget
        self.history_budgets = []
        self.historical_adslot1_rates = []  
        self.historical_adslot2_rates = []
        self.historical_adslot3_rates = []

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
        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0

        history_xi = [result[:, 0] for result in historyAuctionResult]
        # history_slot = [result[:, 1] for result in historyAuctionResult]
        if len(history_xi) > 0:
            last_slot = historyAuctionResult[-1][:, 1]
            # value_counts_slot = np.array
            unique_values, counts = np.unique(np.asanyarray(last_slot), return_counts=True)
            slot_dict = dict(zip(unique_values, counts))
            last_slot1_rate = slot_dict.get(1, 0)/len(last_slot) if len(last_slot) > 0 else 0
            last_slot2_rate = slot_dict.get(2, 0)/len(last_slot) if len(last_slot) > 0 else 0
            last_slot3_rate = slot_dict.get(3, 0)/len(last_slot) if len(last_slot) > 0 else 0
            self.historical_adslot1_rates.append(last_slot1_rate)
            self.historical_adslot2_rates.append(last_slot2_rate)
            self.historical_adslot3_rates.append(last_slot3_rate)

        history_cost = [result[:, 2] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_pSigma = [result[:, 1] for result in historyPValueInfo]
        history_exposure = [result[:,0] for result in historyImpressionResult] 
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_means = np.array([np.mean(xi) for xi in history_xi]) if history_xi else np.zeros(1)

        historical_conversion_means = np.array(
            [np.mean(reward) for reward in history_conversion]) if history_conversion else np.zeros(1)

        historical_LeastWinningCost_means = np.array(
            [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else np.zeros(1)

        historical_pValues_means = np.array([np.mean(value) for value in history_pValue]) if history_pValue else np.zeros(1)

        historical_pValueSigma = np.array([np.mean(sigma) for sigma in history_pSigma]) + 1e-6 if history_pSigma else np.zeros(1)

        historical_bid_means = np.array([np.mean(bid) for bid in historyBid]) if historyBid else np.zeros(1)

        historical_pv_nums = np.array([len(bids) for bids in historyBid]) if historyBid else np.zeros(1)
        
        historical_budgets_left = np.array(self.history_budgets) if self.history_budgets else np.zeros(1)
        
        historical_CPA = np.array([np.sum(history_cost_i)/(1e-5+np.sum(history_xi_i)) for history_cost_i, history_xi_i in zip(history_cost, history_xi)]) if history_cost and history_xi else np.zeros(1)

        historical_adslot1_rates = np.array(self.historical_adslot1_rates) if self.historical_adslot1_rates else np.zeros(1)
        historical_adslot2_rates = np.array(self.historical_adslot2_rates) if self.historical_adslot2_rates else np.zeros(1)
        historical_adslot3_rates = np.array(self.historical_adslot3_rates) if self.historical_adslot3_rates else np.zeros(1)
    
        historical_exposure_rates = np.array([np.mean(exposure) for exposure in history_exposure]) if history_exposure else np.zeros(1)

        historical_CPA_rate = self.cpa / (1e-5 + historical_CPA)
        history_informations = torch.as_tensor(np.nan_to_num(np.array([historical_xi_means, 
                                                                       historical_conversion_means, 
                                                                       historical_LeastWinningCost_means, 
                                                                       historical_pValues_means,
                                                                       historical_pValueSigma, 
                                                                       historical_bid_means, 
                                                                       historical_pv_nums, 
                                                                       historical_budgets_left, 
                                                                       historical_CPA,
                                                                       historical_adslot1_rates,
                                                                       historical_adslot2_rates,
                                                                       historical_adslot3_rates,
                                                                       historical_exposure_rates,
                                                                       historical_CPA_rate,
                                                                       ]), nan=0.0))

        history_informations = history_informations.transpose(1, 0).unsqueeze(1).type(torch.float32)  # 这样就变成了 [T, 1, D] 维度的向量

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)
        current_pValueSigma_mean = np.mean(pValueSigmas + 1e-6)
        current_informations = torch.tensor([
            time_left,
            budget_left,
            current_pValues_mean /1e-3,
            current_pValueSigma_mean / 1e-5,
            current_pv_num/10000,
            self.cpa/50,
            self.budget/1000,
            self.category / 10,
        ]).reshape(1, 1, -1).type(torch.float32) # 这样就变成了 [1, 1, D] 维度的向量
        
        self.history_budgets.append(budget_left)

        ## 假定后面有一个attention 模型，把 history_informations 和 current_informations 输入进去，得到一个 attention 向量
        attention_vector = self.attention.get_attention_vector(history_informations, current_informations,0)

        state = torch.cat([current_informations, attention_vector], dim=-1).squeeze(0)
        # 得到当前状态
        # state = torch.cat([current_informations, attention_vector], dim=-1)

        # 标准化
        # 转换为tenso
        # state = (state - self.normalize_params['state_mean']) / self.normalize_params['state_std']

        # state = torch.tensor(state, dtype=torch.float)

        alpha = self.model(state).view(-1)
        alpha = alpha.cpu().numpy() * 50
        bids = alpha * pValues

        return bids
