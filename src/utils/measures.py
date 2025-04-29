import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from typing import (
    List, Dict, Tuple, 
    Union, Optional
)

from utils.agent_utils import TransitionsFrequencyMatrix, ProxyDict

class OrderStrings :

    @staticmethod
    def order_as_float(list_of_strings: List[str]):
        some_no_floats = False
        list_of_floats = list()
        for x in list_of_strings:
            try:
                x = int(x)
            except:
                try:
                    x = float(x)
                except:
                    some_no_floats = True
            list_of_floats.append(x)
        if not some_no_floats:
            list_of_floats.sort()
        list_of_floats = [str(x) for x in list_of_floats]
        return list_of_floats
            
    @staticmethod
    def dict_as_numeric(list_of_strings: List[str]):
        dict_order = {}
        for x in list_of_strings:
            x_string = str(x)
            try:
                dict_order[x] = int(x_string)
            except:
                try:
                    dict_order[x] = float(x_string)
                except:
                    dict_order[x] = x
        return dict_order

    @staticmethod
    def add_number_if_repeated(name:str, names:List[str]) -> str:
        if name not in names:
            return name
        counter = 1
        while f'{name}_{counter}' in names:
            counter += 1
        return f'{name}_{counter}'


class PPT :
    
    @staticmethod
    def concat_dfs(df1:pd.DataFrame, df2:pd.DataFrame) -> pd.DataFrame:
        # Create dataframe copies
        df1 = df1.copy()
        df2 = df2.copy()
        # Check duplicated columns
        df1 = PPT.drop_duplicates(df1)
        df2 = PPT.drop_duplicates(df2)
        # Dictionary with correct column names
        standard_columns = ['model', 'threshold', 'num_agents', 'id_sim', 'id_player', 'round', 'decision', 'score']
        # Rename columns
        df1 = PPT.rename_columns(df1)
        df2 = PPT.rename_columns(df2)
        # Drop extra columns
        df1 = df1[standard_columns]
        df2 = df2[standard_columns]
        return pd.concat([df1, df2], ignore_index=True)

    @staticmethod
    def drop_duplicates(df:pd.DataFrame) -> pd.DataFrame:
        drop_columns = list()
        if 'room' in df.columns and 'group' in df.columns:
            drop_columns.append('room')
        if 'choice' in df.columns and 'decision' in df.columns:
            drop_columns.append('choice')
        if 'player' in df.columns and 'id_player' in df.columns:
            drop_columns.append('player')
        df.drop(columns=drop_columns, inplace=True)
        return df

    @staticmethod
    def rename_columns(df:pd.DataFrame) -> pd.DataFrame:
        all_dict = {
            'room': 'id_sim',
            'group': 'id_sim',
            'player': 'id_player',
            'num_players': 'num_agents',
            'choice': 'decision'
        }
        # Rename columns in df1
        rename_dict = {
            key:value for key, value in all_dict.items() 
                if key in df.columns and value not in df.columns
        }
        df.rename(columns=rename_dict, inplace=True)        
        return df

    @staticmethod
    def get_group_column(columns: List[str]) -> str:
        if 'id_sim' in columns:
            return 'id_sim'
        elif 'room' in columns:
            return 'room'
        elif 'group' in columns:
            return 'group'
        else:
            raise Exception(f'Error: No column data found. Should be one of "id_sim", "room", or "group".\nColumns found: {columns}')

    @staticmethod
    def get_player_column(columns: List[str]) -> str:
        if 'id_player' in columns:
            return 'id_player'
        elif 'player' in columns:
            return 'player'
        else:
            raise Exception(f'Error: No player data found. Should be one of "id_player" or "player".\nColumns found: {columns}')

    @staticmethod
    def get_num_player_column(columns: List[str]) -> str:
        if 'num_players' in columns:
            return 'num_players'
        elif 'num_agents' in columns:
            return 'num_agents'
        else:
            raise Exception(f'Error: No number of players column found. Should be one of "num_players" or "num_agents".\nColumns found: {columns}')

    @staticmethod
    def get_decision_column(columns: List[str]) -> str:
        if 'decision' in columns:
            return 'decision'
        elif 'choice' in columns:
            return 'choice'
        else:
            raise Exception(f'Error: No decision data found. Should be one of "decision" or "choice".\nColumns found: {columns}')

    @staticmethod
    def get_fixed_parameters(data: pd.DataFrame) -> List[int]:
        assert('threshold' in data.columns)
        num_players_col = PPT.get_num_player_column(data.columns)
        pairs = data[[num_players_col, 'threshold']].dropna().values.tolist()
        pairs = [tuple(x) for x in pairs]
        pairs = list(set(pairs))
        list_fixed = list()
        for num_p, threshold in pairs:
            fixed_params = {
                'num_agents': int(num_p),
                'threshold': threshold
            }
            list_fixed.append(fixed_params)
        # Run checks
        for fixed_parameters in list_fixed:
            num_ag = int(fixed_parameters["num_agents"])
            threshold = fixed_parameters["threshold"]
            num_agent_column = PPT.get_num_player_column(data.columns)
            try:
                df = data.groupby([num_agent_column, "threshold"]).get_group(tuple([num_ag, threshold]))
            except Exception as e:
                print(num_ag, threshold, pairs)
                raise Exception(e)
        return list_fixed
    
    @staticmethod
    def get_number_of_groups(data: pd.DataFrame) -> List[int]:
        num_players_col = PPT.get_num_player_column(data.columns)
        group_column = PPT.get_group_column(data.columns)
        list_sizes = list()
        for key, grp in data.groupby([num_players_col, 'threshold']):
            size = len(grp[group_column].unique())
            group_size = {'num_episodes':size}
            list_sizes.append(group_size)
        return list_sizes


class PathUtils :
        
    @staticmethod
    def add_file_name(
                path: Path, 
                file_name: str,
                extension: str
            ) -> Path:
        file = Path.joinpath(path, f'{file_name}.{extension}')
        counter = 0
        while file.exists():
            counter += 1
            file = Path.joinpath(path, f'{file_name}_{counter}.{extension}')
        return file


class ConditionalEntropy :
    '''
    Calculates the conditional entropy values
    for each of a number of simulations of a model
    '''

    def __init__(
                self, 
                data: pd.DataFrame,
                T: Optional[int]=20
            ) -> None:
        self.data = data
        self.T = T
        self.group_column = PPT.get_group_column(self.data.columns)
        self.decision_column = PPT.get_decision_column(self.data.columns)
        self.debug = False
        np.seterr(divide = 'ignore') 

    def get_entropy(self) -> List[float]:
        M = self.data['round'].unique().max()
        df = pd.DataFrame(self.data[self.data['round'] >= (M - self.T)]).reset_index(drop=True)
        entropy = list()
        for group, df in df.groupby(self.group_column):
            entropy.append(self.get_group_entropy(df))
        return entropy

    def get_group_entropy(
                self,
                df: pd.DataFrame
            ) -> float:
        tm = self.get_group_states(df)
        ce = self.calculate_entropy(tm)
        return ce

    def get_group_states(
                self,
                df: pd.DataFrame
            ) -> ProxyDict:
        # Get the number of agents in group
        num_agents = self.get_df_num_agents(df)
        # Create states followed by group
        states = list()
        for round_, round_data in df.groupby('round'):
            state = round_data[self.decision_column].values
            assert(len(state) == num_agents), f'{round_=}\n{round_data}'
            states.append(tuple(state))
        # Create a proxy dictionary
        all_states = list(product([0,1], repeat=num_agents))
        tm = ProxyDict(
            keys=all_states,
            initial_val=0
        )
        for state in states:
            tm.increment(state)
        return tm
    
    def calculate_entropy(
                self,
                tm: ProxyDict,
                rate: Optional[bool]=True
            ) -> float:
        tm.normalize()
        H = 0
        probs = tm.as_array()
        log_probs = np.log2(probs)
        log_probs[np.isinf(log_probs)] = 0
        prob_logs = np.multiply(probs, log_probs)
        H = -sum(prob_logs)
        if self.debug:
            print('States Relative Frequency:')
            print(tm)
            print(f'{prob_logs=}')
            print('H:', H)
        if rate:
            N = np.log2(len(tm))
            H /= N
        return(H)
        
    def get_contidional_entropy(self) -> List[float]:
        M = self.data['round'].unique().max()
        df = pd.DataFrame(self.data[self.data['round'] >= (M - self.T)]).reset_index(drop=True)
        conditional_entropy = list()
        for group, df in df.groupby(self.group_column):
            conditional_entropy.append(self.get_group_conditional_entropy(df))
        return conditional_entropy

    def get_group_conditional_entropy(
                self,
                df: pd.DataFrame
            ) -> float:
        tm = self.get_group_transitions(df)
        ce = self.calculate_conditional_entropy(tm)
        return ce

    def get_group_transitions(
                self,
                df: pd.DataFrame
            ) -> TransitionsFrequencyMatrix:
        # Get the number of agents in group
        num_agents = self.get_df_num_agents(df)
        # Create states followed by group
        states = list()
        for round_, round_data in df.groupby('round'):
            state = round_data[self.decision_column].values
            states.append(state)
        # Create transition dataframe
        df_transitions = pd.DataFrame({
            'state': states,
            'next_state': states[1:] + [np.nan]
        }).dropna()
        df_transitions['transition'] = df_transitions[['state', 'next_state']].apply(lambda x: tuple((tuple(x['state']), tuple(x['next_state']))), axis=1)
        # Create Transition frequency matrix
        tm = TransitionsFrequencyMatrix(
            num_agents=num_agents,
            uniform=False
        )
        for transition in df_transitions['transition'].values:
            tm.increment(transition)
        return tm
    
    def calculate_conditional_entropy(
                self,
                tm: TransitionsFrequencyMatrix,
                rate: Optional[bool]=True
            ) -> float:
        H = 0
        A = tm.trans_freqs
        # Find conditional probabilities
        row_sums = np.sum(A, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Prevent division by zero
        cond_probs = A / row_sums
        log_cond_probs = np.log2(cond_probs)
        log_cond_probs[np.isinf(log_cond_probs)] = 0
        # Find joint probabilities
        joint_probs = A / np.sum(A)
        # Calculate conditional entropy
        prob_logs = np.multiply(joint_probs, log_cond_probs)
        H = -sum(prob_logs.flatten())
        if rate:
            N = np.log2(len(tm))
            H /= N
        return(H)

    def get_df_num_agents(
                self, 
                df: pd.DataFrame
            ) -> int:
        num_groups = len(df[self.group_column].unique())
        assert(num_groups == 1)
        player_column = PPT.get_player_column(self.data.columns)
        num_agents = len(df[player_column].unique())
        return num_agents


class Fourier :

    def __init__(
                self,
                data: pd.DataFrame,
                T: Optional[int]=20
            ) -> None:
        self.data = data
        self.T = T
        self.debug = False

    def get_fourier(self) -> List[float]:
        M = self.data['round'].unique().max()
        df = pd.DataFrame(self.data[self.data['round'] >= (M - self.T)]).reset_index(drop=True)
        try:
            fourier = list()
            group_column = PPT.get_group_column(self.data.columns)
            for key, grp in df.groupby(group_column):
                f_max = Fourier.get_group_max_fourier(grp)
                fourier.append(f_max)
        except:
            fourier = Fourier.get_group_max_fourier(df)
        return fourier

    @staticmethod
    def get_group_max_fourier(df: pd.DataFrame) -> float:                  
        attendances = Fourier.get_round_attendance(df)
        states = Fourier.get_states(attendances)
        fourier_max = Fourier.get_fourier_max(states)
        return fourier_max

    @staticmethod
    def get_round_attendance(rnd_data:pd.DataFrame) -> pd.Series:
        players = rnd_data['id_player'].unique()
        first_p = players[0]
        pl_data = rnd_data.groupby('id_player').get_group(first_p)
        return pl_data['attendance']

    @staticmethod
    def get_states(attendance:pd.Series) -> np.ndarray:
        states = np.zeros(attendance.shape[0])
        for i, state_ in enumerate(attendance):
            state = json.loads(state_)
            states[i] = int("".join(str(x) for x in state), 2)
        return states

    @staticmethod
    def get_fourier_max(states:np.ndarray) -> np.ndarray:
        fourier = np.fft.fft(states)
        fourier = [x.real for x in fourier]
        return max(fourier)


class GetMeasurements :
    
    def __init__(
                self,
                data: pd.DataFrame,
                measures: List[str],
                normalize: Optional[bool]=False,
                T: Optional[int]=20,
                per_round: Optional[bool]=False,
                per_player: Optional[bool]=False
            ) -> None:
        #-----------------------------
        # Book keeping
        #-----------------------------
        for measure in measures:
            assert(measure in ['attendance', 'efficiency', 'normalized_efficiency', 'inequality', 'entropy', 'conditional_entropy', 'fourier', 'round_efficiency']), f'Error: {measure} not in measures'
        self.measures = measures
        self.normalize = normalize
        self.T = T
        self.data = data
        #-----------------------------
        # Find columns to groupby		
        #-----------------------------
        self.group_column = PPT.get_group_column(self.data.columns)
        self.num_players_column = PPT.get_num_player_column(self.data.columns)
        columns = ['model', 'treatment', 'threshold', self.group_column, self.num_players_column]
        if per_round:
            columns.append('round')
        self.per_round = per_round
        if per_player:
            self.player_column = PPT.get_player_column(self.data.columns)
            columns.append(self.player_column)
        self.columns = [c for c in columns if c in self.data.columns]
        #-----------------------------
        # Keep only T last rounds
        #-----------------------------
        self.keep_last_rounds()

    def get_measurements(self) -> pd.DataFrame:
        if self.per_round:
            return self.get_measurements_per_round()
        else:
            return self.get_measurements_all()

    def get_measurements_per_round(self) -> pd.DataFrame:
        init = True
        for measure in self.measures:
            variable = self.get_variable_from_measure(measure)
            fun = getattr(GetMeasurements, measure)
            if init:
                list_df = list()
                print('====>', self.columns)
                for key, grp in self.data.groupby(self.columns):
                    print('>>>>>')
                    print(fun(grp))
                    df = pd.DataFrame({variable: fun(grp)})
                    df[self.columns] = key
                    list_df.append(df)
                df = pd.concat(list_df, ignore_index=True)
                if self.normalize:
                    df[variable] = (df[variable]-df[variable].mean())/df[variable].std()
                init = False
            else:
                list_df = list()
                for key, grp in self.data.groupby(self.columns):
                    aux = pd.DataFrame({variable: fun(grp)})
                    aux[self.columns] = key
                    list_df.append(df)
                aux = pd.concat(list_df, ignore_index=True)
                if self.normalize:
                    aux[variable] = (aux[variable]-aux[variable].mean())/aux[variable].std()
                df = pd.merge(df, aux, on=self.columns, how='inner')
        return df			

    def get_measurements_all(self) -> pd.DataFrame:
        assert(len(self.measures) > 0)
        init = True
        for measure in self.measures:
            fun = eval(f'GetMeasurements.{measure}')
            if init:
                df = self.data.groupby(self.columns).apply(fun).reset_index()
                df.rename(columns={0:measure}, inplace=True)
                if self.normalize:
                    df[measure] = (df[measure]-df[measure].mean())/df[measure].std()
                init = False
            else:
                aux = self.data.groupby(self.columns).apply(fun).reset_index()
                aux.rename(columns={0:measure}, inplace=True)
                if self.normalize:
                    aux[measure] = (aux[measure]-aux[measure].mean())/aux[measure].std()
                df = pd.merge(df, aux, on=self.columns, how='inner')
        return df			

    def get_variable_from_measure(self, measure:str) -> str:
        if measure == 'round_efficiency':
            return 'efficiency'
        if measure == 'round_conditional_entropy':
            return 'conditional_entropy'
    
    def keep_last_rounds(self) -> None:
        df_list = list()
        for key, grp in self.data.groupby(self.group_column):
            num_rounds = max(grp["round"].unique())
            grp_aux = pd.DataFrame(grp[grp['round'] >= (num_rounds - self.T)]).reset_index(drop=True)
            df_list.append(grp_aux)
        self.data = pd.concat(df_list, ignore_index=True)

    @staticmethod
    def attendance(df: pd.DataFrame) -> float:
        # assert(GetMeasurements.one_group_only(df))
        decision_column = PPT.get_decision_column(df.columns)
        return df[decision_column].mean()

    @staticmethod
    def efficiency(df: pd.DataFrame) -> float:
        # assert(GetMeasurements.one_group_only(df))
        return df['score'].mean()

    @staticmethod
    def normalized_efficiency(df: pd.DataFrame) -> float:
        # assert(GetMeasurements.one_group_only(df))
        threshold = df['threshold'].mean()
        if threshold == 0:
            threshold = 1e-3
        return df['score'].mean() / threshold

    @staticmethod
    def round_efficiency(df: pd.DataFrame) -> float:
        # assert(GetMeasurements.one_group_only(df))
        return df.groupby('round')['score'].mean()#.reset_index()

    @staticmethod
    def inequality(df: pd.DataFrame) -> float:
        # assert(GetMeasurements.one_group_only(df))
        player_column = PPT.get_player_column(df.columns)
        mean_scores = df.groupby(player_column)['score'].mean().reset_index()
        # mean_scores['score'] = (mean_scores['score'] + 1) / 2
        return mean_scores['score'].std()

    @staticmethod
    def entropy(df: pd.DataFrame) -> float:
        assert(GetMeasurements.one_group_only(df))
        ge = ConditionalEntropy(df, T=np.infty)
        return ge.get_group_entropy(df)

    @staticmethod
    def conditional_entropy(df: pd.DataFrame) -> float:
        # assert(GetMeasurements.one_group_only(df))
        ge = ConditionalEntropy(df, T=np.infty)
        return ge.get_group_conditional_entropy(df)

    @staticmethod
    def fourier(df: pd.DataFrame) -> float:
        # assert(GetMeasurements.one_group_only(df))
        return Fourier.get_group_max_fourier(df)

    @staticmethod
    def one_group_only(df: pd.DataFrame) -> bool:
        group_column = PPT.get_group_column(df.columns)
        groups = df[group_column].unique()
        return len(groups) == 1
    

class Grid:
    '''Creates a most compact grid and ravels and unravels'''

    def __init__(
                self, 
                length:int, 
                vertical:Optional[bool]=True,
                iterable:Optional[bool]=False
            ) -> None:
        num1 = int(np.sqrt(length))
        num2 = length // num1
        if num1 * num2 < length:
            self.shape = (max(num1, num2) + 1, min(num1, num2))
        else:
            self.shape = (max(num1, num2), min(num1, num2))
        if not vertical:
            self.shape = (self.shape[1], self.shape[0])
        self.rows = self.shape[0]
        self.cols = self.shape[1]
        self.iterable = iterable

    def index_to_pair(self, i):
        if self.shape[0] == 1 or self.shape[1] == 1:
            if self.iterable:
                return i
            else:
                return [i]
        return np.unravel_index(i, self.shape)
    
    def pair_to_index(self, x, y):
        return np.ravel_multi_index((x, y), self.shape)