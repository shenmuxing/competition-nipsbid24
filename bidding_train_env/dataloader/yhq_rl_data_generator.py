# yhq_rl_data_generator.py 
# 用于将原始数据转换为适合RL模型训练的数据格式

import os
import pandas as pd
import numpy as np
import warnings
import glob
import gc
from utils.zjy_config import get_config

warnings.filterwarnings('ignore')
import time   #用于计算耗时
from numba import jit
from utils.logger import setup_logger

def calculate_state(row, budget, remainingBudget, timeStepIndex, timeStepIndexNum):
    timeleft = (timeStepIndexNum - timeStepIndex) / timeStepIndexNum
    bgtleft = remainingBudget / budget if budget > 0 else 0

    state = (
        timeleft, bgtleft,
        # row.bid if not np.isnan(row.bid) else 0, 
        row.bid_lag_1, row.bid_lag_2, row.bid_lag_3,
        row.leastWinningCost_lag_1, row.leastWinningCost_lag_2, row.leastWinningCost_lag_3,
        row.pValue, row.pValue_lag_1, row.pValue_lag_2, row.pValue_lag_3,
        row.conversionAction_lag_1, row.conversionAction_lag_2, row.conversionAction_lag_3,
        row.xi_lag_1, row.xi_lag_2, row.xi_lag_3,
        row.cumulative_realCost, row.cumulative_conversions,row.cumulative_pValue,
        budget, row.CPAConstraint, row.advertiserCategoryIndex
    )

    return state

class RlDataGenerator:
    """
    RL Data Generator for RL models.
    Reads raw data and constructs training data suitable for reinforcement learning.
    """

    def __init__(self, file_folder_path="./data/traffic", **kwargs):
        # 读取可能的配置
        if not hasattr(kwargs, 'config'):
            self.config = get_config()
        else:
            self.config = kwargs['config']
        
        self.file_folder_path = file_folder_path
        ##################################### Runtime info 测试路径
        self.training_data_path = self.file_folder_path + "/" + "yhq_training_data_rlData_folder"
        # self.training_data_path = self.file_folder_path + "/" + 'myself' + "/" + "training_data_rlData_folder"

    def _remove_duplicates(self, df):
        return df # 暂时取消去重，因为这一步不会有太大的收益，反而花费很多时间进行处理
    
    def batch_generate_rl_data(self):
        os.makedirs(self.training_data_path, exist_ok=True)
        
        is_csv = self.config['IS_DOCKER']
        file_extension = '*.csv' if is_csv else '*.feather'
        files = glob.glob(os.path.join(self.file_folder_path, file_extension))
        
        print(files)
        
        for file_path in files:
            print("开始处理文件：", file_path)
            
            df = pd.read_csv(file_path) if is_csv else pd.read_feather(file_path)

            df_processed = self._generate_rl_data(df)
            
            file_name = os.path.basename(file_path)
            output_file_name = file_name.replace('.csv' if is_csv else '.feather', '-yhq_rlData' + ('.csv' if is_csv else '.feather'))
            output_path = os.path.join(self.training_data_path, output_file_name)
            
            if is_csv:
                df_processed.to_csv(output_path, index=False)
            else:
                df_processed.to_feather(output_path)
            
            # training_data_list.append(df_processed)
            del df, df_processed
            gc.collect()
            print("处理文件成功：", file_path)


    def _generate_rl_data(self, df):
        # 对 DataFrame 进行初始排序
        df = df.sort_values(by=['advertiserNumber', 'timeStepIndex', 'pvIndex'])

        # 计算 realCost
        df['realCost'] = df['cost'] * df['isExposed']

        # 提前计算聚合数据
        agg_functions = {
            'bid': 'mean',
            'leastWinningCost': 'mean',
            'conversionAction': 'mean',
            'xi': 'mean',
            'pValue': 'mean',
            'isExposed': 'sum',
            'cost': 'mean',
            'realCost': 'sum',
        }
        
        aggregated_data = df.groupby(['advertiserNumber', 'timeStepIndex']).agg(agg_functions).reset_index()

        # 计算历史窗口特征
        window_size = 3
        for col in ['bid', 'leastWinningCost', 'conversionAction', 'xi', 'pValue', 'cost']:
            for i in range(window_size):
                aggregated_data[f'{col}_lag_{i+1}'] = aggregated_data.groupby('advertiserNumber')[col].shift(i+1).fillna(0)

        # 计算累计值
        aggregated_data['cumulative_realCost'] = aggregated_data.groupby('advertiserNumber')['realCost'].cumsum().shift(1).fillna(0)
        aggregated_data['cumulative_conversions'] = aggregated_data.groupby('advertiserNumber')['conversionAction'].cumsum().shift(1).fillna(0)
        aggregated_data['cumulative_pValue'] = aggregated_data.groupby('advertiserNumber')['pValue'].cumsum().shift(1).fillna(0)

        # 计算总的 realCost 和 pValue
        total_stats = aggregated_data.groupby('advertiserNumber').agg({
            'realCost': 'sum',
            'pValue': 'sum',
            'conversionAction': 'sum'
        }).rename(columns={
            'realCost': 'total_realCost',
            'pValue': 'total_pValue',
            'conversionAction': 'total_conversions'
        })
        
        aggregated_data = pd.merge(aggregated_data, total_stats, on='advertiserNumber')

        training_data_rows = []

        for (deliveryPeriodIndex, advertiserNumber, advertiserCategoryIndex, budget, CPAConstraint), group in df.groupby(
            ['deliveryPeriodIndex', 'advertiserNumber', 'advertiserCategoryIndex', 'budget', 'CPAConstraint']):

            # 在每个 group 内进行去重操作
            done_rows = group[(group['isEnd'] == 1) | (group['timeStepIndex'] == 47)]
            if not done_rows.empty:
                min_done_row = done_rows.iloc[0]
                other_rows = group[~((group['isEnd'] == 1) | (group['timeStepIndex'] == 47))]
                group = pd.concat([other_rows, min_done_row.to_frame().T], ignore_index=True)
                
            # 对处理后的 group 再次排序
            group = group.sort_values(by=['timeStepIndex', 'pvIndex'])

            # 合并聚合数据
            group = pd.merge(group, aggregated_data, on=['advertiserNumber', 'timeStepIndex'], suffixes=('', '_agg'))

            for row in group.itertuples(index=False):
                timeStepIndex = row.timeStepIndex
                timeStepIndexNum = 48
                remainingBudget = row.remainingBudget

                state = calculate_state(row, budget, remainingBudget, timeStepIndex, timeStepIndexNum)
                action = row.bid / row.pValue if (row.pValue > 0 and not np.isnan(row.bid)) else 0
                reward = row.conversionAction if row.isExposed == 1 else 0
                reward_continuous = row.pValue if row.isExposed == 1 else 0
                done = 1 if timeStepIndex == timeStepIndexNum - 1 or row.isEnd == 1 else 0

                # 计算额外的奖励项
                if done == 1:
                    total_reward = row.total_conversions
                    total_reward_continuous = row.total_pValue
                    total_cost = row.total_realCost
                    C = CPAConstraint
                    total_cpa = total_cost / total_reward if total_reward > 0 else 0

                    actual_total_reward = min((C / total_cpa) ** 2, 1) * total_reward
                    delta_reward = actual_total_reward - total_reward
                    delta_reward_continuous = (actual_total_reward / total_reward) * total_reward_continuous - total_reward_continuous if total_reward > 0 else 0

                    reward += delta_reward
                    reward_continuous += delta_reward_continuous

                training_data_rows.append({
                    'deliveryPeriodIndex': deliveryPeriodIndex,
                    'advertiserNumber': advertiserNumber,
                    'advertiserCategoryIndex': advertiserCategoryIndex,
                    'budget': budget,
                    'CPAConstraint': CPAConstraint,
                    'realAllCost': row.total_realCost,
                    'realAllConversion': row.total_conversions,
                    'timeStepIndex': timeStepIndex,
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'reward_continuous': reward_continuous,
                    'done': done
                })

        training_data = pd.DataFrame(training_data_rows)
        # 计算next_state

        # 为所有列创建聚合函数
        all_columns = [column for column in training_data.columns if column not in ['advertiserNumber','timeStepIndex']]
        agg_functions = {col: 'first' for col in all_columns}  # 默认使用 'first' 函数

        # 为特定列指定特殊的聚合函数
        special_agg_functions = {
            'state': lambda x: tuple(np.mean(x.tolist(), axis=0)),
            'action': 'mean',
            'reward': 'mean',
            'reward_continuous': 'mean',
            'done': 'max'
        }

        # 更新 agg_functions
        agg_functions.update(special_agg_functions)

        # 进行聚合
        aggregated_data = training_data.groupby(['advertiserNumber', 'timeStepIndex']).agg(agg_functions).reset_index()

        # 计算聚合数据的next_state
        aggregated_data['next_state'] = aggregated_data.groupby('advertiserNumber')['state'].shift(-1)
        aggregated_data.loc[aggregated_data['done'] == 1, 'next_state'] = None

        # 将聚合数据的next_state合并到原始数据中
        training_data = pd.merge(
            training_data,
            aggregated_data[['advertiserNumber', 'timeStepIndex', 'next_state']],
            on=['advertiserNumber', 'timeStepIndex'],
            how='left'
        )

        return training_data.reset_index(drop=True)


def generate_rl_data():
    
    config = get_config() 
    config.log_file = "logs/zjy/yhq_rl_data_generator.log"
    logger = setup_logger(cfg = config)


    file_folder_path = "./data/traffic"
    data_loader = RlDataGenerator(file_folder_path=file_folder_path,config = config)
    data_loader.batch_generate_rl_data()

if __name__ == '__main__':
    generate_rl_data()
