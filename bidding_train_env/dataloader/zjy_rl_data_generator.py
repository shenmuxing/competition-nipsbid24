import os
import pandas as pd
import warnings
import glob
import numpy as np
from utils.config import get_config

warnings.filterwarnings('ignore')

augment_rate = 3 # 扩充训练数据至原来的augment_rate+1倍
mean_of_action = 81.77
std_of_action = 41.22
min_of_action = 0
max_of_action = 163.54

fake_pv_rate = 0.7 # 随机生成pValue 和 pvIndex的比例

mean_of_pv_index = 10415
std_of_pv_index = 5800
min_of_pv_index = 1000
max_of_pv_index = 28888

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
        self.training_data_path = self.file_folder_path + "/" + "zjy_training_data_rlData_folder"


    def batch_generate_rl_data(self):
        os.makedirs(self.training_data_path, exist_ok=True)
        
        is_csv = self.config['IS_DOCKER']
        file_extension = '*.csv' if is_csv else '*.feather'
        files = glob.glob(os.path.join(self.file_folder_path, file_extension))
        
        print(files)
        training_data_list = []
        
        for file_path in files:
            print("开始处理文件：", file_path)
            
            df = pd.read_csv(file_path) if is_csv else pd.read_feather(file_path)
            df_processed = self._generate_rl_data(df)
            
            file_name = os.path.basename(file_path)
            output_file_name = file_name.replace('.csv' if is_csv else '.feather', '-rlData' + ('.csv' if is_csv else '.feather'))
            output_path = os.path.join(self.training_data_path, output_file_name)
            
            if is_csv:
                df_processed.to_csv(output_path, index=False)
            else:
                df_processed.to_feather(output_path)
            
            training_data_list.append(df_processed)
            del df, df_processed
            print("处理文件成功：", file_path)
        
        combined_dataframe = pd.concat(training_data_list, axis=0, ignore_index=True)
        combined_file_name = "training_data_all-rlData" + ('.csv' if is_csv else '.feather')
        combined_file_path = os.path.join(self.training_data_path, combined_file_name)
        
        if is_csv:
            combined_dataframe.to_csv(combined_file_path, index=False)
        else:
            combined_dataframe.to_feather(combined_file_path)
        
        print("整合多天训练数据成功；保存至:", combined_file_path)

    def _generate_rl_data(self, df):
        """
        Construct a DataFrame in reinforcement learning format based on the raw data.

        Args:
            df (pd.DataFrame): The raw data DataFrame.

        Returns:
            pd.DataFrame: The constructed training data in reinforcement learning format.
        """

        training_data_rows = []

        # Step 1: Compute max_bid_without_me and second_bid_without_me
        df = df.sort_values(by=['timeStepIndex', 'pvIndex']).reset_index(drop=True)

        df['action'] = np.where(df['pValue'] == 0, 0, df['bid'] / df['pValue'])

        group_keys = ['timeStepIndex', 'advertiserNumber']

        grouped = df.groupby(group_keys)

        df['mean_action'] = grouped['action'].transform('mean')
        df['mean_isEnd'] = grouped['isEnd'].transform('mean')

        grouped = df.groupby(['timeStepIndex'])

        df['mean_remainingBudget'] = grouped['remainingBudget'].transform('mean')
        df['mean_CPAConstraint'] = grouped['CPAConstraint'].transform('mean')

        df['mean_pValue'] = grouped['pValue'].transform('mean')
        df['quantile1_pValue'] = grouped['pValue'].transform(lambda x: x.quantile(0.75))
        df['quantile2_pValue'] = grouped['pValue'].transform(lambda x: x.quantile(0.25))
        df['std_pValue'] = grouped['pValue'].transform('std')

        df["mean_pValueSigma"] = grouped["pValueSigma"].transform("mean")

        
        # 定义一个函数，用于获取每组的前N个唯一值（降序）
        def get_top_n_unique(series, n):
            return series.drop_duplicates().nlargest(n).tolist()

        # 优化 'mean_action' 的计算
        # 一次性获取每个 'timeStepIndex' 下的前3个唯一的 'mean_action' 值
        top_actions = df.groupby('timeStepIndex')['mean_action'].apply(lambda x: get_top_n_unique(x, 3)).reset_index()

        # 将列表展开成多列
        top_actions[['max_action', 'second_action', 'third_action']] = top_actions['mean_action'].apply(pd.Series)
        top_actions.drop(columns=['mean_action'], inplace=True)

        # 合并回原始 DataFrame
        df = df.merge(top_actions, on='timeStepIndex', how='left')

        df['max_action_without_me'] = df['max_action']
        df.loc[df['mean_action'] == df['max_action'], 'max_action_without_me'] = df['second_action']

        # 2.4 计算 second_bid_without_me
        # 如果当前行的 bid != max_bid，则 second_bid_without_me = second_bid
        # 否则，second_bid_without_me = third_bid
        df['second_action_without_me'] = df['second_action']
        df.loc[df['mean_action'] == df['second_action'], 'second_action_without_me'] = df['third_action']
        
        group_keys =  ['timeStepIndex', 'pvIndex']

        # grouped = df.groupby(group_keys)

        N = 5  # 获取每组的前 5 个唯一 'bid'

        # 1. 按分组键和 'bid' 进行降序排序
        df_sorted = df.sort_values(by=group_keys + ['bid'], ascending=[True, True, False])

        # 2. 删除每组内重复的 'bid'，保留每个 'bid' 的第一个出现（即最大的 'bid'）
        df_unique = df_sorted.drop_duplicates(subset=group_keys + ['bid'])

        # 3. 按分组键分组，并为每组获取前 N 个 'bid'
        top_bids = df_unique.groupby(group_keys).head(N)

        # 如果需要将结果重塑为每组一个列表，可以进一步处理
        top_bids = top_bids.groupby(group_keys)['bid'].apply(list).reset_index()


        # 将列表展开成多列，并命名
        top_bids[['max_bid', 'second_bid', 'third_bid', 'fourth_bid', 'fifth_bid']] = top_bids['bid'].apply(pd.Series)
        top_bids.drop(columns=['bid'], inplace=True)

        # 合并回原始 DataFrame
        df = df.merge(top_bids, on=group_keys, how='left')

        # df['max_bid_without_me'] = df['max_bid']
        # df.loc[df['bid'] == df['max_bid'], 'max_bid_without_me'] = df['second_bid']
        bid = df['bid'].values
        max_bid = df['max_bid'].values
        second_bid = df['second_bid'].values
        third_bid = df['third_bid'].values
        fourth_bid = df['fourth_bid'].values
        fifth_bid = df['fifth_bid'].values
        max_bid_without_me = np.where(bid == max_bid, second_bid, max_bid)
        second_bid_without_me = np.where(bid == second_bid, third_bid, second_bid)
        third_bid_without_me = np.where(bid == third_bid, fourth_bid, third_bid)
        fourth_bid_without_me = np.where(bid == fourth_bid, fifth_bid, fourth_bid)
        df[['max_bid_without_me', 'second_bid_without_me', 'third_bid_without_me', 'fourth_bid_without_me']] = np.column_stack([
            max_bid_without_me,
            second_bid_without_me,
            third_bid_without_me,
            fourth_bid_without_me
        ])

        # Sort df appropriately
        df = df.sort_values(by=['advertiserNumber', 'timeStepIndex', 'pvIndex']).reset_index(drop=True)

        for (
                deliveryPeriodIndex, advertiserNumber, advertiserCategoryIndex, budget,
                CPAConstraint), group in df.groupby(
            ['deliveryPeriodIndex', 'advertiserNumber', 'advertiserCategoryIndex', 'budget', 'CPAConstraint']):

            group = group.sort_values('timeStepIndex')

            # Compute realAllCost and realAllConversion
            # realAllCost = (group['isExposed'] * group['cost']).sum()
            # realAllConversion = group['conversionAction'].sum()

            # Initialize lists to store history features
            # historical_xi_means = []
            historical_conversion_means = []
            historical_LeastWinningCost_means = []
            historical_pValues_means = []
            historical_bid_means = []
            historical_pv_nums = []
            historical_budgets_left = []
            historical_CPA = []
            historical_pValueSigma = []
            historical_adslot1_rates = []
            historical_adslot2_rates = []
            historical_adslot3_rates = []
            historical_exposure_rates = []
            historical_CPA_rate = []
            historical_cost = []
            # historical_actions = []

            total_reward = 0
            total_reward_continuous = 0

            # For each timeStepIndex
            time_steps = sorted(group['timeStepIndex'].unique())
            timeStepIndexNum = 48

            for idx, timeStepIndex in enumerate(time_steps):
                current_timeStep_data = group[group['timeStepIndex'] == timeStepIndex]

                # Determine done
                done = int(timeStepIndex == timeStepIndexNum - 1 or current_timeStep_data['isEnd'].iloc[0] == 1)
                is_last_step = int(timeStepIndex == timeStepIndexNum - 1)

                # Compute current_informations
                time_left = (timeStepIndexNum - timeStepIndex) / timeStepIndexNum
                remaining_budget = current_timeStep_data['remainingBudget'].iloc[0]
                budget_left = remaining_budget / budget if budget > 0 else 0
                total_budget = budget
                current_pValues_mean = current_timeStep_data['pValue'].mean()
                current_pValues_quantile1 = current_timeStep_data['pValue'].quantile(0.75)
                current_pValues_quantile2 = current_timeStep_data['pValue'].quantile(0.25)
                current_pValues_std = current_timeStep_data['pValue'].std()
                current_pv_num = len(current_timeStep_data) 
                current_CPA = ((current_timeStep_data['isExposed'] * current_timeStep_data['cost']).sum())/(1e-5 + current_timeStep_data[current_timeStep_data['isExposed'] == 1]['conversionAction'].sum())
                current_pValueSigma_mean = (current_timeStep_data['pValueSigma'].mean() + 1e-6)

                mean_CPAConstraint = current_timeStep_data['mean_CPAConstraint'].mean()
                mean_remainingBudget = current_timeStep_data['mean_remainingBudget'].mean()
                

                current_informations = [time_left, 
                                        budget_left, 
                                        current_pValues_mean / 1e-3 , 
                                        current_pValues_quantile1 / 1e-3, 
                                        current_pValues_quantile2 / 1e-3, 
                                        current_pValues_std / 1e-3, 
                                        current_pValueSigma_mean / 1e-5, 
                                        current_pv_num / 10000,
                                        CPAConstraint /50,
                                        total_budget / 1000,
                                        int(advertiserCategoryIndex ==0),
                                        int(advertiserCategoryIndex ==1),
                                        int(advertiserCategoryIndex ==2),
                                        int(advertiserCategoryIndex ==3),
                                        int(advertiserCategoryIndex ==4),
                                        done,
                                        is_last_step,
                                        ]

                # Build history_informations
                history_informations = [
                    historical_conversion_means.copy(),
                    historical_LeastWinningCost_means.copy(),
                    historical_pValues_means.copy(),
                    historical_pValueSigma.copy(),
                    historical_bid_means.copy(),
                    historical_pv_nums.copy(),
                    historical_budgets_left.copy(),
                    historical_CPA.copy(),
                    historical_adslot1_rates.copy(),
                    historical_adslot2_rates.copy(),
                    historical_adslot3_rates.copy(),
                    historical_exposure_rates.copy(),
                    historical_CPA_rate.copy(),
                    historical_cost.copy(),
                    # historical_actions.copy(),
                ]

                # Compute action
                total_bid = current_timeStep_data['bid'].sum()
                total_value = current_timeStep_data['pValue'].sum()
                action = total_bid / total_value if total_value > 0 else 0
                max_bid_without_me_mean = current_timeStep_data['max_bid_without_me'].mean()
                second_bid_without_me_mean = current_timeStep_data['second_bid_without_me'].mean()
                max_action_without_me = current_timeStep_data['max_action_without_me'].mean()
                second_action_without_me = current_timeStep_data['second_action_without_me'].mean()
                
                # Compute reward
                reward = (current_timeStep_data[current_timeStep_data['isExposed'] == 1]['conversionAction'].sum())*min((CPAConstraint/(1e-5+current_CPA))**3,1) -\
                       (current_timeStep_data[current_timeStep_data['isExposed'] == 0]['pValue'].sum())/8
                reward_continuous = (current_timeStep_data[current_timeStep_data['isExposed'] == 1]['pValue'].sum())*min((CPAConstraint/(1e-5+current_CPA))**3,1) -\
                      (current_timeStep_data[current_timeStep_data['isExposed'] == 0]['pValue'].sum())/8

                total_reward += reward
                total_reward_continuous += reward_continuous
                
                others_current_pValues_mean = current_timeStep_data['mean_pValue'].mean()
                others_current_pValues_quantile1 = current_timeStep_data['quantile1_pValue'].mean()
                others_current_pValues_quantile2 = current_timeStep_data['quantile2_pValue'].mean()
                others_current_pValues_std = current_timeStep_data['std_pValue'].mean()
                others_current_pValueSigma_mean = (current_timeStep_data['mean_pValueSigma'].mean() + 1e-6)

                current_state_others = [
                                        mean_CPAConstraint/50,
                                        mean_remainingBudget/1000,
                                        others_current_pValues_mean / 1e-3,
                                        others_current_pValues_quantile1 / 1e-3,
                                        others_current_pValues_quantile2 / 1e-3,
                                        others_current_pValues_std / 1e-3,
                                        others_current_pValueSigma_mean / 1e-5,
                                        ]


                if not is_last_step:
                    # Next time step exists
                    next_timeStepIndex = time_steps[idx + 1]
                    next_timeStep_data = group[group['timeStepIndex'] == next_timeStepIndex]

                    # Determine done
                    next_done = int(next_timeStepIndex == timeStepIndexNum - 1 or next_timeStep_data['isEnd'].iloc[0] == 1)
                    next_is_last_step = int(next_timeStepIndex == timeStepIndexNum - 1)

                    # Compute next_state_current_information
                    next_time_left = (timeStepIndexNum - next_timeStepIndex) / timeStepIndexNum
                    next_budget_left = next_timeStep_data['remainingBudget'].iloc[0] / budget if budget > 0 else 0
                    next_pValues_mean = next_timeStep_data['pValue'].mean()
                    next_pValues_quantile1 = next_timeStep_data['pValue'].quantile(0.75)
                    next_pValues_quantile2 = next_timeStep_data['pValue'].quantile(0.25)
                    next_pValues_std = next_timeStep_data['pValue'].std()
                    next_pValueSigma_mean = (next_timeStep_data['pValueSigma'].mean() + 1e-6)
                    next_pv_num = len(next_timeStep_data)
                    
                    # next_mean_action = next_timeStep_data['mean_action'].mean()
                    next_mean_CPAConstraint = next_timeStep_data['mean_CPAConstraint'].mean()
                    next_mean_remainingBudget = next_timeStep_data['mean_remainingBudget'].mean()
                    # next_mean_isEnd = next_timeStep_data['mean_isEnd'].mean()

                    next_state_current_information = [next_time_left, 
                                                      next_budget_left, 
                                                      next_pValues_mean / 1e-3,
                                                      next_pValues_quantile1 / 1e-3, 
                                                      next_pValues_quantile2 / 1e-3, 
                                                      next_pValues_std / 1e-3, 
                                                      next_pValueSigma_mean / 1e-5,                     
                                                      next_pv_num / 10000,
                                                      CPAConstraint / 50,
                                                      total_budget/1000,
                                                      int(advertiserCategoryIndex ==0),
                                                      int(advertiserCategoryIndex ==1),
                                                      int(advertiserCategoryIndex ==2),
                                                      int(advertiserCategoryIndex ==3),
                                                      int(advertiserCategoryIndex ==4),
                                                      next_done,
                                                      next_is_last_step,
                                                      ]
                    
                    others_next_pValues_mean = next_timeStep_data['mean_pValue'].mean()
                    others_next_pValues_quantile1 = next_timeStep_data['quantile1_pValue'].mean()
                    others_next_pValues_quantile2 = next_timeStep_data['quantile2_pValue'].mean()
                    others_next_pValues_std = next_timeStep_data['std_pValue'].mean()
                    others_next_pValueSigma_mean = (next_timeStep_data['mean_pValueSigma'].mean() + 1e-6)
                    next_state_current_state_others = [
                                                    next_mean_CPAConstraint/50,
                                                    next_mean_remainingBudget/1000,
                                                    others_next_pValues_mean / 1e-3,
                                                    others_next_pValues_quantile1 / 1e-3,
                                                    others_next_pValues_quantile2 / 1e-3,
                                                    others_next_pValues_std / 1e-3,
                                                    others_next_pValueSigma_mean / 1e-5,
                                                    ]

                    # Update historical data for next state (include current time step data)

                    conversion_mean = current_timeStep_data['conversionAction'].mean() if pd.notnull(current_timeStep_data['conversionAction'].mean()) else 0
                    leastWinningCost_mean = current_timeStep_data['leastWinningCost'].mean() if pd.notnull(current_timeStep_data['leastWinningCost'].mean()) else 0
                    pValue_mean = current_timeStep_data['pValue'].mean() if pd.notnull(current_timeStep_data['pValue'].mean()) else 0
                    bid_mean = current_timeStep_data['bid'].mean() if pd.notnull(current_timeStep_data['bid'].mean()) else 0
                    pv_num = len(current_timeStep_data)
                    budget_left = budget_left  # already computed
                    adslot_counts = current_timeStep_data['adSlot'].value_counts()
                    current_adslot1_rate = adslot_counts.get(1, 0)/current_pv_num if current_pv_num > 0 else 0
                    current_adslot2_rate = adslot_counts.get(2, 0)/current_pv_num if current_pv_num > 0 else 0
                    current_adslot3_rate = adslot_counts.get(3, 0)/current_pv_num if current_pv_num > 0 else 0
                    exposure_rate = current_timeStep_data['isExposed'].mean() if pd.notnull(current_timeStep_data['isExposed'].mean()) else 0

                    total_cost = (current_timeStep_data['cost'] * current_timeStep_data['isExposed']).sum()
                    next_historical_conversion_means = historical_conversion_means + [conversion_mean]
                    next_historical_LeastWinningCost_means = historical_LeastWinningCost_means + [leastWinningCost_mean]
                    next_historical_pValues_means = historical_pValues_means + [pValue_mean]
                    next_historical_bid_means = historical_bid_means + [bid_mean]
                    next_historical_pv_nums = historical_pv_nums + [pv_num]
                    next_historical_budgets_left = historical_budgets_left + [budget_left]
                    next_historical_CPA = historical_CPA + [current_CPA]
                    next_historical_pValueSigma = historical_pValueSigma + [current_pValueSigma_mean]
                    next_historical_adslot1_rates = historical_adslot1_rates + [current_adslot1_rate]
                    next_historical_adslot2_rates = historical_adslot2_rates + [current_adslot2_rate]
                    next_historical_adslot3_rates = historical_adslot3_rates + [current_adslot3_rate]
                    next_historical_exposure_rates = historical_exposure_rates + [exposure_rate]
                    next_historical_CPA_rate = historical_CPA_rate + [CPAConstraint/(1e-5+current_CPA)]
                    next_historical_cost = historical_cost + [total_cost]
                    # next_historical_actions = historical_actions + [action]

                    next_state_history_information = [
                        # next_historical_xi_means.copy(),
                        next_historical_conversion_means.copy(),
                        next_historical_LeastWinningCost_means.copy(),
                        next_historical_pValues_means.copy(),
                        next_historical_pValueSigma.copy(),
                        next_historical_bid_means.copy(),
                        next_historical_pv_nums.copy(),
                        next_historical_budgets_left.copy(),
                        next_historical_CPA.copy(),
                        next_historical_adslot1_rates.copy(),
                        next_historical_adslot2_rates.copy(),
                        next_historical_adslot3_rates.copy(),
                        next_historical_exposure_rates.copy(),
                        next_historical_CPA_rate.copy(),
                        next_historical_cost.copy(),
                        # next_historical_actions.copy(),
                    ]
                else:

                    # No next state
                    next_state_current_information = None
                    next_state_history_information = None
                    next_state_current_state_others = None

                # Store data
                training_data_rows.append({
                    'deliveryPeriodIndex': deliveryPeriodIndex,
                    'advertiserNumber': advertiserNumber,
                    'advertiserCategoryIndex': advertiserCategoryIndex,
                    'budget': budget,
                    'CPAConstraint': CPAConstraint,
                    'timeStepIndex': timeStepIndex,
                    'state_current_informations': current_informations,
                    'state_history_informations': history_informations,
                    'state_current_others': current_state_others,
                    'action': action,
                    'reward': reward,
                    'real_cost': total_cost,
                    'reward_continuous': reward_continuous,
                    'done': done,
                    'is_last_step': is_last_step,                    
                    'next_state_current_information': next_state_current_information,
                    'next_state_history_information': next_state_history_information,
                    'next_state_current_others': next_state_current_state_others,
                    'max_bid_without_me': max_bid_without_me_mean,
                   'second_bid_without_me': second_bid_without_me_mean,
                   'max_action_without_me': max_action_without_me,
                   'second_action_without_me': second_action_without_me,
                   'true_data' : 1,
                   'conversion_mean': conversion_mean,
                })

                for _ in range(augment_rate):
                    # 再创造一些新数据
                    random_numbers = np.random.normal(mean_of_action, std_of_action)
                    action_new = np.ones(current_timeStep_data['pValue'].shape[0])* np.clip(random_numbers, min_of_action, max_of_action)
                    bids_new =  (current_timeStep_data['pValue'] * action_new ).to_numpy()
                    max_bid_without_me = current_timeStep_data['max_bid_without_me'].to_numpy()
                    second_bid_without_me = current_timeStep_data['second_bid_without_me'].to_numpy()
                    third_bid_without_me = current_timeStep_data['third_bid_without_me'].to_numpy()
                    fourth_bid_without_me = current_timeStep_data['fourth_bid_without_me'].to_numpy()

                    cost_new = np.zeros_like(bids_new)
                    is_exposed_new = np.zeros_like(bids_new)  

                    index_first_bid = (bids_new > max_bid_without_me)
                    cost_new[index_first_bid] = max_bid_without_me[index_first_bid]
                    is_exposed_new[index_first_bid] = 1
                    num_adslot1 = np.sum(index_first_bid)
                    # 否则，当bids_new > second_bid_without_me时，此时的cost为second_bid_without_me
                    index_second_bid = (bids_new > second_bid_without_me) & (bids_new <= max_bid_without_me)    
                    cost_new[index_second_bid] = second_bid_without_me[index_second_bid]
                    is_exposed_new[index_second_bid] = (np.random.rand(index_second_bid.sum()) < 0.8).astype(int)
                    num_adslot2 = np.sum(index_second_bid)
                    # 否则如果, 当bids_new > leastWinningCost_mean时，此时的cost为leastWinningCost_mean
                    index_least_cost = (second_bid_without_me >= bids_new) & ((bids_new ) > third_bid_without_me)
                    cost_new[index_least_cost] = third_bid_without_me[index_least_cost]
                    is_exposed_new[index_least_cost] = (np.random.rand(index_least_cost.sum()) < 0.6).astype(int)
                    num_adslot3 = np.sum(index_least_cost)

                    # real_all_cost_new = (is_exposed_new * cost_new).sum()
                    # conversion_new = np.random.binomial(n = len(is_exposed_new),p = is_exposed_new * current_timeStep_data['pValue'].to_numpy())
                    is_conversion = np.random.random(len(is_exposed_new))
                    is_conversion = (is_conversion < current_timeStep_data['pValue'].to_numpy()).astype(int)
                    conversion_new = is_conversion * is_exposed_new
                    conversion_mean_new = conversion_new.mean()
                    # if conversion_mean_new == conversion_mean:
                    #     print('conversion_mean_new == conversion_mean')
                    real_cost_new = (is_exposed_new * cost_new).sum()

                    remaining_budget_new = current_timeStep_data['remainingBudget'].iloc[0] - real_cost_new

                    if  remaining_budget_new < 0:
                        real_cost_cumsum = (is_exposed_new * cost_new).cumsum()
                        mask_keep = remaining_budget_new + real_cost_cumsum > 0

                        cost_new = np.zeros_like(bids_new)
                        is_exposed_new = np.zeros_like(bids_new)

                        index_first_bid = index_first_bid & mask_keep
                        cost_new[index_first_bid] = max_bid_without_me[index_first_bid]
                        is_exposed_new[index_first_bid] = 1
                        num_adslot1 = np.sum(index_first_bid)

                        index_second_bid = index_second_bid & mask_keep
                        cost_new[index_second_bid] = second_bid_without_me[index_second_bid]
                        is_exposed_new[index_second_bid] = (np.random.rand(index_second_bid.sum()) < 0.8).astype(int)
                        num_adslot2 = np.sum(index_second_bid)

                        index_least_cost = index_least_cost & mask_keep
                        cost_new[index_least_cost] = third_bid_without_me[index_least_cost]
                        is_exposed_new[index_least_cost] = (np.random.rand(index_least_cost.sum()) < 0.6).astype(int)
                        num_adslot3 = np.sum(index_least_cost)
                        
                        conversion_new = is_conversion * is_exposed_new
                        conversion_mean_new = conversion_new.mean()
                        real_cost_new = (is_exposed_new * cost_new).sum()
                        remaining_budget_new = current_timeStep_data['remainingBudget'].iloc[0] - real_cost_new
                        
                    current_CPA_new = ((real_cost_new)/(1e-5 +(is_exposed_new * current_timeStep_data['pValue'].to_numpy()).sum()))

                    reward_continuous_new = (is_exposed_new * current_timeStep_data['pValue']).sum()*min((CPAConstraint/(1e-5+current_CPA_new))**3,1)  - (current_timeStep_data[is_exposed_new == 0]['pValue'].sum())/8

                    reward_new = round(reward_continuous_new + np.random.normal(0,0.001),2)
                    total_cost_new = real_cost_new


                    if not is_last_step:
                        next_done_new = int(next_done or (remaining_budget_new < 0.1)) 

                        # 当bids_new > max_bid_without_me时，此时的cost为max_bid_without_me
                        current_adslot1_rate_new = num_adslot1 / current_pv_num if current_pv_num > 0 else 0
                        current_adslot2_rate_new = num_adslot2 / current_pv_num if current_pv_num > 0 else 0
                        current_adslot3_rate_new = num_adslot3 / current_pv_num if current_pv_num > 0 else 0
                        exposure_rate_new = is_exposed_new.mean() if pd.notnull(is_exposed_new.mean()) else 0
                        total_cost_all_agents_time = (max_bid_without_me[index_first_bid]).sum() + second_bid_without_me[~index_first_bid].sum() + (second_bid_without_me[index_second_bid | index_first_bid]).sum() * 0.8 \
                                +(third_bid_without_me[~(index_second_bid|index_first_bid)]).sum() * 0.8 + (third_bid_without_me[index_least_cost|index_second_bid|index_first_bid]).sum() * 0.6 \
                                    + (fourth_bid_without_me[~(index_least_cost|index_second_bid|index_first_bid)]).sum() * 0.6

                        next_mean_remainingBudget_new = current_timeStep_data['mean_remainingBudget'].mean()- total_cost_all_agents_time/48
                        next_historical_conversion_means_new = historical_conversion_means + [conversion_mean_new]
                        next_historical_LeastWinningCost_means_new = historical_LeastWinningCost_means + [leastWinningCost_mean]
                        next_historical_pValues_means_new = historical_pValues_means + [pValue_mean]
                        next_historical_bid_means_new = historical_bid_means + [bids_new.mean()]
                        next_historical_pv_nums_new = historical_pv_nums + [pv_num]
                        next_historical_budgets_left_new = historical_budgets_left + [budget_left]
                        next_historical_CPA_new = historical_CPA + [current_CPA_new]
                        next_historical_pValueSigma_new = historical_pValueSigma + [current_pValueSigma_mean]
                        next_historical_adslot1_rates_new = historical_adslot1_rates + [current_adslot1_rate_new]
                        next_historical_adslot2_rates_new = historical_adslot2_rates + [current_adslot2_rate_new]
                        next_historical_adslot3_rates_new = historical_adslot3_rates + [current_adslot3_rate_new]
                        next_historical_exposure_rates_new = historical_exposure_rates + [exposure_rate_new]
                        next_historical_CPA_rate_new = historical_CPA_rate + [CPAConstraint/(1e-5+current_CPA_new)]
                        next_historical_cost_new = historical_cost + [total_cost_new]
                        # next_historical_actions_new = historical_actions + [action_new.mean()]

                        next_state_history_information = [
                            next_historical_conversion_means_new.copy(),
                            next_historical_LeastWinningCost_means_new.copy(),
                            next_historical_pValues_means_new.copy(),
                            next_historical_pValueSigma_new.copy(),
                            next_historical_bid_means_new.copy(),
                            next_historical_pv_nums_new.copy(),
                            next_historical_budgets_left_new.copy(),
                            next_historical_CPA_new.copy(),
                            next_historical_adslot1_rates_new.copy(),
                            next_historical_adslot2_rates_new.copy(),
                            next_historical_adslot3_rates_new.copy(),
                            next_historical_exposure_rates_new.copy(),
                            next_historical_CPA_rate_new.copy(),
                            next_historical_cost_new.copy(),
                            # next_historical_actions_new.copy(),
                        ]
                        if np.random.rand() < fake_pv_rate:
                            # 随机生成假的pv
                            next_pv_num_tmp = round(np.clip(np.random.normal(mean_of_pv_index, std_of_pv_index), min_of_pv_index, max_of_pv_index))
                            next_pValues_tmp = np.clip(np.random.normal(next_pValues_mean, next_pValues_std,size = (next_pv_num_tmp,)),0,1)
                            next_pValues_mean_tmp = next_pValues_tmp.mean()
                            next_pValues_quantile1_tmp = np.percentile(next_pValues_tmp, 75, interpolation='linear')
                            next_pValues_quantile2_tmp = np.percentile(next_pValues_tmp, 25, interpolation='linear')
                            next_pValues_std_tmp = next_pValues_tmp.std()
                            next_pValueSigma_mean_tmp = next_pValueSigma_mean + np.random.normal(0,0.00002)
                        else:
                            # 真实的pv
                            next_pv_num_tmp = next_pv_num
                            next_pValues_mean_tmp = next_pValues_mean
                            next_pValues_quantile1_tmp = next_pValues_quantile1 
                            next_pValues_quantile2_tmp = next_pValues_quantile2 
                            next_pValues_std_tmp = next_pValues_std
                            next_pValueSigma_mean_tmp = next_pValueSigma_mean

                        next_state_current_information = [
                            next_time_left,
                            remaining_budget_new / budget if budget > 0 else 0,
                            next_pValues_mean_tmp / 1e-3,
                            next_pValues_quantile1_tmp / 1e-3, 
                            next_pValues_quantile2_tmp / 1e-3, 
                            next_pValues_std_tmp / 1e-3, 
                            next_pValueSigma_mean_tmp / 1e-5,
                            next_pv_num_tmp / 10000,
                            CPAConstraint / 50,
                            total_budget/1000,
                            int(advertiserCategoryIndex ==0),
                            int(advertiserCategoryIndex ==1),
                            int(advertiserCategoryIndex ==2),
                            int(advertiserCategoryIndex ==3),
                            int(advertiserCategoryIndex ==4),
                            next_done_new,
                            next_is_last_step,
                        ]
                        next_state_current_state_others = [
                                                next_mean_CPAConstraint/50,
                                                next_mean_remainingBudget_new/1000,
                                                others_next_pValues_mean/1e-3,
                                                others_next_pValues_quantile1/1e-3,
                                                others_next_pValues_quantile2/1e-3,
                                                others_next_pValues_std/1e-3,
                                                others_next_pValueSigma_mean/1e-5,
                                                ]

                    else:
                        # No next state
                        # next_remaining_budget_new = current_timeStep_data['remainingBudget'].iloc[0]
                        # next_remaining_budget_new -= total_cost_new
                        # reward_new -= next_remaining_budget_new/CPAConstraint/3
                        # reward_continuous_new -= next_remaining_budget_new/CPAConstraint/3
              
                        next_state_current_information = None
                        next_state_history_information = None
                        next_state_current_state_others = None

                    training_data_rows.append({
                        'deliveryPeriodIndex': deliveryPeriodIndex,
                        'advertiserNumber': advertiserNumber,
                        'advertiserCategoryIndex': advertiserCategoryIndex,
                        'budget': budget,
                        'CPAConstraint': CPAConstraint,
                        'timeStepIndex': timeStepIndex,
                        'state_current_informations': current_informations,
                        'state_current_others': current_state_others,
                        'state_history_informations': history_informations,
                        'action': action_new.mean(),
                        'reward': reward_new,
                        'real_cost': total_cost_new,
                        'reward_continuous': reward_continuous_new,
                        'done': done,
                        'is_last_step': is_last_step,
                        'next_state_current_information': next_state_current_information,
                        'next_state_history_information': next_state_history_information,
                        'next_state_current_others': next_state_current_state_others,
                        'max_bid_without_me': max_bid_without_me_mean,
                        'second_bid_without_me': second_bid_without_me_mean,
                        'max_action_without_me': max_action_without_me,
                        'second_action_without_me': second_action_without_me,
                        'true_data':0,
                        'conversion_mean':conversion_mean_new,
                    })

                # After storing, update the historical data for the next iteration
                # xi_mean = current_timeStep_data['xi'].mean() if pd.notnull(current_timeStep_data['xi'].mean()) else 0
                conversion_mean = current_timeStep_data['conversionAction'].mean() if pd.notnull(current_timeStep_data['conversionAction'].mean()) else 0
                leastWinningCost_mean = current_timeStep_data['leastWinningCost'].mean() if pd.notnull(current_timeStep_data['leastWinningCost'].mean()) else 0
                pValue_mean = current_timeStep_data['pValue'].mean() if pd.notnull(current_timeStep_data['pValue'].mean()) else 0
                bid_mean = current_timeStep_data['bid'].mean() if pd.notnull(current_timeStep_data['bid'].mean()) else 0
                pv_num = len(current_timeStep_data)

                # historical_xi_means.append(xi_mean)
                historical_conversion_means.append(conversion_mean)
                historical_LeastWinningCost_means.append(leastWinningCost_mean)
                historical_pValues_means.append(pValue_mean)
                historical_bid_means.append(bid_mean)
                historical_pv_nums.append(pv_num)
                historical_budgets_left.append(budget_left)
                historical_CPA.append(current_CPA)
                historical_pValueSigma.append(current_pValueSigma_mean)
                historical_adslot1_rates.append(current_adslot1_rate)
                historical_adslot2_rates.append(current_adslot2_rate)
                historical_adslot3_rates.append(current_adslot3_rate)
                historical_exposure_rates.append(exposure_rate)
                historical_CPA_rate.append(CPAConstraint/(1e-5+current_CPA))
                historical_cost.append(total_cost)
                # historical_actions.append(action)


        # Build the DataFrame after processing all time steps for this group
        training_data = pd.DataFrame(training_data_rows)
        training_data = training_data.sort_values(by=['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex'])

        return training_data

def generate_rl_data():

    config = get_config() 
    file_folder_path = "./data/traffic"
    data_loader = RlDataGenerator(file_folder_path=file_folder_path,config = config)
    data_loader.batch_generate_rl_data()

if __name__ == '__main__':
    generate_rl_data()
