import numpy as np
import math
import logging
from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv

from utils.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def run_test():
    """
    offline evaluation
    """
    config = get_config()
    logger.info(f'Config: {config}')
    if config['IS_DOCKER']:
        file_path = './data/traffic/period-7.csv'
    else:
        file_path = './data/traffic/period-7.feather'

    data_loader = TestDataLoader(file_path=file_path)
    env = OfflineEnv()
    # agent = PlayerBiddingStrategy()
    # print(agent.name)

    keys, test_dict = data_loader.keys, data_loader.test_dict
    exist_category = set()
    overall_reward = 0
    overall_score = 0
    for key in keys:
        category_i = key[2]
        if len(exist_category) == 6:
            break
        elif category_i not in exist_category:
            exist_category.add(category_i)
        else:
            continue
        logger.info(f'Key: {key}')
        num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts, budgt_i, cpa_i = data_loader.mock_data(key)   # timeStepIndex 对应值，list元素ndarray
        agent = PlayerBiddingStrategy(budget=budgt_i,cpa = cpa_i)
        rewards = np.zeros(num_timeStepIndex)
        history = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': [],
            'historyLeastWinningCost': [],
            'historyPValueInfo': []
        }

        for timeStep_index in range(num_timeStepIndex):
            logger.info(f'Timestep Index: {timeStep_index + 1} Begin')

            pValue = pValues[timeStep_index]
            pValueSigma = pValueSigmas[timeStep_index]
            leastWinningCost = leastWinningCosts[timeStep_index]

            if agent.remaining_budget < env.min_remaining_budget:
                bid = np.zeros(pValue.shape[0])
            else:

                bid = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                    history["historyBids"],
                                    history["historyAuctionResult"], history["historyImpressionResult"],
                                    history["historyLeastWinningCost"])

            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                        leastWinningCost)

            # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            while over_cost_ratio > 0:
                pv_index = np.where(tick_status == 1)[0]
                dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                    replace=False)
                bid[dropped_pv_index] = 0
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                            leastWinningCost)
                over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

            agent.remaining_budget -= np.sum(tick_cost)
            rewards[timeStep_index] = np.sum(tick_conversion)
            temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
            history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
            history["historyBids"].append(bid)
            history["historyLeastWinningCost"].append(leastWinningCost)
            temAuctionResult = np.array(
                [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
            history["historyAuctionResult"].append(temAuctionResult)
            temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
            history["historyImpressionResult"].append(temImpressionResult)
            logger.info(f'Timestep Index: {timeStep_index + 1} End')
        all_reward = np.sum(rewards)
        all_cost = agent.budget - agent.remaining_budget
        cpa_real = all_cost / (all_reward + 1e-10)
        cpa_constraint = agent.cpa
        score = getScore_nips(all_reward, cpa_real, cpa_constraint)
        overall_reward += all_reward
        overall_score += score
        agent.reset()
        logger.info(f"Category: {category_i}")
        logger.info(f"Budget: {budgt_i}")
        logger.info(f'Total Reward: {all_reward}')
        logger.info(f'Total Cost: {all_cost}')
        logger.info(f'CPA-real: {cpa_real}')
        logger.info(f'CPA-constraint: {cpa_constraint}')
        logger.info(f'Score: {score}')
        logger.info(f"bid history: {[np.mean(i) for i in history['historyBids']]}")
    
    logger.info(f"Overall Reward: {overall_reward}")
    logger.info(f"Overall Score: {overall_score}")

if __name__ == '__main__':
    run_test()
