import numpy as np
import pandas as pd
import statsmodels.api as sm

from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from itertools import product
from typing import Optional, Union, Dict, List
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils.measures import GetMeasurements
from utils.cherrypick_simulations import CherryPickEquilibria
from config import PATHS

class AlternationIndex:
    '''Estimates the alternation index from simulated data'''

    def __init__(
                self, 
                num_points: Optional[int]=20,
                num_episodes: Optional[int]=20,
                max_agents: Optional[int]=8, 
                max_epsilon: Optional[float]=0.025,
                seed: Optional[Union[int, None]]=None
            ) -> None:
        self.num_points = num_points
        self.max_agents = max_agents
        self.max_epsilon = max_epsilon
        self.num_episodes = num_episodes
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.configuration_points = self.create_configurations()
        # self.measures = ['normalized_efficiency', 'entropy', 'conditional_entropy', 'inequality']
        self.measures = ['normalized_efficiency', 'inequality']
        self.data = None
        self.sklearn_coefficients = None
        self.statsmodels_coefficients = None
        self.coefficients = None
        self.index_path = PATHS['index_path']
        self.priority = 'statsmodels'
        self.debug = True

    def __call__(self, df:pd.DataFrame) -> float:
        '''Calculate the index from the dataframe'''
        if self.coefficients is None:
            self.create_index_calculator()
        assert('normalized_efficiency' in df.columns)
        # assert('entropy' in df.columns)
        # assert('conditional_entropy' in df.columns)
        assert('inequality' in df.columns)
        # Extract intercept and weights
        intercept = self.coefficients[0]
        weights = np.array(self.coefficients[1:])
        # Compute linear combination
        linear_combination = np.dot(df[self.measures], weights) + intercept
        # Sigmoid function
        probabilities = 1 / (1 + np.exp(-linear_combination))
        return probabilities
        
    def create_index_calculator(self) -> None:
        '''Create the index calculator based on the priority'''
        if self.priority == 'sklearn':
            if self.sklearn_coefficients is None:
                self.create_index_sklearn()
            self.coefficients = self.sklearn_coefficients
        elif self.priority == 'statsmodels':
            if self.statsmodels_coefficients is None:
                self.create_index_statsmodels()
            self.coefficients = self.statsmodels_coefficients
        else:
            raise ValueError('Priority must be sklearn or statsmodels')        
        
    def create_index_sklearn(self) -> Dict[str, float]:
        if self.data is None:
            # Create dataframe with all data
            df = self.simulate_data()
            self.data = df
        else:
            df = self.data
        # Create target variable
        # 1 for alternation, 0 for segmentation/random
        df['target'] = df['data_type'].apply(lambda x: 1 if x == 'alternation' else 0)
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            df[self.measures], 
            df['target'], 
            test_size=0.2,
            random_state=0
        )
        # Fit logistic regression
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        # Predict and evaluate
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))
        df_index = pd.DataFrame({
            'measure': ['intercept'] + self.measures,
            'coefficient': clf.intercept_.tolist() + clf.coef_[0].tolist()
        })
        self.sklearn_coefficients = df_index
        # Save to file
        index_path = self.index_path / Path('sklearn_coefficients.csv')
        df_index.to_csv(index_path, index=False)
        if self.debug:
            print('Saved sklearn coefficients to', index_path)
        return df_index
    
    def create_index_statsmodels(self) -> Dict[str, float]:
        if self.data is None:
            # Create dataframe with all data
            df = self.simulate_data()
            self.data = df
        else:
            df = self.data
        # Create target variable
        # 1 for alternation, 0 for segmentation/random
        df['target'] = df['data_type'].apply(lambda x: 1 if x == 'alternation' else 0)
        # Fit logistic regression
        X = sm.add_constant(df[self.measures])
        y = df['target']
        model = sm.Logit(y, X)
        result = model.fit(disp=0)
        print(result.summary())
        df_index = pd.DataFrame({
            'measure': ['intercept'] + self.measures,
            'coefficient': result.params.tolist()
        })
        self.statsmodels_coefficients = df_index
        # Save to file
        index_path = self.index_path / Path('statsmodels_coefficients.csv')
        df_index.to_csv(index_path, index=False)
        if self.debug:
            print('Saved statsmodels coefficients to', index_path)
        return df_index

    def simulate_data(self) -> pd.DataFrame:        
        data_types = ['alternation', 'segmentation', 'random']
        df_list = list()
        for data_type in data_types:
            df = self.simulate_data_kind(data_type)
            df['data_type'] = data_type
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
        df['num_agents'] = df['num_agents'].astype(int)
        df['threshold'] = df['threshold'].astype(float)
        self.data = df
        return df
    
    def simulate_data_kind(self, data_type:str) -> pd.DataFrame:
        assert(data_type in ['segmentation', 'alternation', 'random'])
        num_episodes = deepcopy(self.num_episodes)
        if data_type in ['segmentation', 'random']:
            num_episodes = self.num_episodes // 2
        df_list = list()
        for num_agents, threshold, epsilon in tqdm(self.configuration_points, desc=f'Running configurations for {data_type}'):
            eq_generator = CherryPickEquilibria(
                num_agents=int(num_agents),
                threshold=threshold,
                epsilon=epsilon,
                num_episodes=num_episodes,
                seed=self.seed
            )
            eq_generator.debug = False
            df_alternation = eq_generator.generate_data(data_type)
            get_m = GetMeasurements(
                data=df_alternation, 
                measures=self.measures,
                normalize=False,
            )
            df = get_m.get_measurements() 
            df['epsilon'] = epsilon
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
        return df

    @staticmethod
    def from_file(priority:Optional[str]='sklearn'):
        '''Load the index from file'''
        index = AlternationIndex()
        index_path = PATHS['index_path']
        if priority == 'sklearn':
            index_path = index_path / Path('sklearn_coefficients.csv')
            df = pd.read_csv(index_path)
            index.sklearn_coefficients = df['coefficient'].values
        elif priority == 'statsmodels':
            index_path = index_path / Path('statsmodels_coefficients.csv')
            df = pd.read_csv(index_path)
            index.statsmodels_coefficients = df['coefficient'].values
        else:
            raise ValueError('Priority must be sklearn or statsmodels')
        index.priority = priority
        index.create_index_calculator()
        return index
    
    @staticmethod
    def complete_measures(measures: List[str]) -> List[str]:
        dict_check = AlternationIndex.check_alternation_index_in_measures(measures)
        return dict_check['measures']

    @staticmethod
    def check_alternation_index_in_measures(measures: List[str]) -> Dict[str, any]:
        measures_ = deepcopy(measures)
        if 'alternation_index' in measures_:
            index = measures_.index('alternation_index')
            measures_.pop(index)
            # measures_ += ['normalized_efficiency', 'inequality', 'entropy', 'conditional_entropy']
            measures_ += ['normalized_efficiency', 'inequality']
            measures_ = list(set(measures_))
            check = True
        else:
            check = False
        dict_check = {
            'measures': measures_,
            'check': check 
        }
        return dict_check
    
    def create_configurations(self):
        range_epsilon = np.linspace(0, self.max_epsilon, 10)
        range_num_agents = list(set(np.linspace(2, self.max_agents, 10)))
        pairs = product(range_num_agents, range_epsilon)
        configurations = list()
        for num_agents, epsilon in pairs:
            num_agents = int(num_agents)
            for B in range(1, num_agents):
                triplet = (num_agents, B / num_agents, epsilon)
                configurations.append(triplet)
        if len(configurations) >= self.num_points:
            configurations = self.rng.choice(configurations, size=self.num_points, replace=False)
        else:
            print('Warning: Not enough configurations, using all')
        return configurations