import pandas as pd
from datetime import datetime
import numpy as np

from gym.spaces import Discrete, Box
from gym import Env

def get_data():
    #Get the historical quotes 

    xls = 'historico_cotacoes.xlsx'
    dict_df = pd.read_excel(xls,sheet_name=None)

    del dict_df['LU0122613903']

    # Parse the date and set it as index
    for key in dict_df.keys():
        dict_df[key]['Date'] = dict_df[key].apply(lambda x: pd.to_datetime(x['Date'], infer_datetime_format = True).date(), axis=1)
        dict_df[key].set_index('Date',drop=True, inplace=True)

    # Create the DataFrame from the Dictionary of Dataframes, removing rows with null values
    df_all = pd.concat([dict_df[f_name][['Close']].rename(columns={'Close':f_name}) for f_name in dict_df.keys()],axis=1).sort_index(ascending=False)

    # Remove NaN
    max_date, min_date = df_all.dropna().index.max(), df_all.dropna().index.min()
    df_all = df_all[(df_all.index <= max_date) & (df_all.index >= min_date)].interpolate()


    #Create a DataFrame with the daily growth
    df_grow = df_all.apply(lambda x: x/x.shift(-1), axis=0).dropna().sort_index(ascending=True)
    return df_grow


def get_initial_qty(etf_list):
    return [1.0 for etf in etf_list]

BASE_CONFIG = {
    "SIM_DURATION": 1,  # Simulation interation
    "INITIAL_CASH": 1.0
}

class SimBuyModel:
    def __init__(self, config: dict = None, data: pd.DataFrame = None):
        assert data is not None, 'A Data Frame is required'
        self.data = data
        self.sim_config = BASE_CONFIG.copy()
        if config is not None:
            self.sim_config.update(config)
        self.start = np.random.choice(len(data) - self.sim_config['SIM_DURATION'] - 3)
        self.cash = self.sim_config['INITIAL_CASH']
        self.transact_amount = self.sim_config['INITIAL_CASH'] / self.sim_config['SIM_DURATION']
        self.etf_amount = {etf:1.0 for etf in self.data.columns}
        self.action_to_etf = {i+1:etf for i, etf in enumerate(self.data.columns)} 
        self.idx = self.start
        self.total_reward = 0

    def get_observation(self):
        return np.array(self.data.iloc[self.idx])

    def step(self, action):
        def get_future_valuation(etf, action):
            amount = self.etf_amount[etf]
            if action and (self.action_to_etf[action] == etf):
                amount *= self.data.iloc[self.idx+1][etf]*self.data.iloc[self.idx+2][etf]
                amount += self.transact_amount
                amount *= self.data.iloc[self.idx+1][etf]
            else:
                amount *= self.data.iloc[self.idx+1][etf]*self.data.iloc[self.idx+2][etf]*self.data.iloc[self.idx+3][etf]
            return amount
        
        current_value = sum([self.etf_amount[etf] for etf in self.data.columns]) + self.cash
        if action:
            self.cash -= self.transact_amount
        new_value = sum([get_future_valuation(etf,action) for etf in self.data.columns]) + self.cash
        self.total_reward += new_value - current_value
        self.idx += 1
        self.etf_amount = {etf:self.etf_amount[etf]*self.data.iloc[self.idx][etf] for etf in self.data.columns}
        obs = self.get_observation()
        done = (self.idx - self.start) >= self.sim_config["SIM_DURATION"]
        reward = self.total_reward if done else 0.0
        return obs, reward, done, {}

class SimBuyEnv(Env):

    def __init__(self, config: dict = None):
        self.data = get_data()
        n_etfs = len(self.data.columns)
        self.action_space = Discrete(n_etfs+1)
        self.observation_space = Box(low=np.array([self.data.min().min()]*n_etfs),
                        high=np.array([self.data.max().max()]*n_etfs),
                        dtype=np.float64)
        if config is None or "sim_config" not in config.keys():
            self.sim_config = None
        else:
            self.sim_config = config["sim_config"]

    def reset(self):
        self.sim = SimBuyModel(self.sim_config, self.data)

        # Start processes and initialize resources
        obs = self.sim.get_observation()
        assert self.observation_space.contains(obs), "{} not in {}".format(obs, self.observation_space)
        return obs

    def step(self, action):
        assert action in range(self.action_space.n)

        obs, reward, done, info = self.sim.step(action)

        assert self.observation_space.contains(obs), "{} not in {}".format(obs, self.observation_space)
        return obs, reward, done, info


# Mandatory
class SimBaselineBuy:
    def __init__(self, sim_config: dict = None):
        self.env = SimBuyEnv()

    def get_action(self, obs):
        return np.random.randint(len(self.env.data.columns)+1)

    def run(self):
        done = False
        total_reward = 0
        obs = self.env.reset()
        while not done:
            action = self.get_action(obs)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward

        return total_reward