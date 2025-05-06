'''
Helper functions to gather and process data
'''

import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from time import sleep
from pathlib import Path
from copy import deepcopy
from itertools import product
from random import seed, choices
from IPython.display import clear_output
from typing import List, Dict, Tuple, Union, Optional

from Classes.bar import Bar
from Classes.agentes import Agente

class Episode :
    '''
    Runs the problem for a number of rounds and keeps tally of everything.
    '''

    def __init__(self, environment:Bar, agents:List[Agente], model:str, num_rounds:int):
        '''
        Input:
            - environment, object with the environment on which to test the agents.
            - agents, list with the agents.
            - num_rounds, int with the number of rounds.
        '''
        self.environment = environment
        self.agents = agents
        self.model = model
        self.num_rounds = num_rounds
        self.id = uuid.uuid1()
        self.sleep_time = 0.3
    
    def play_round(self, verbose:Optional[bool]=False):
        '''
        Plays one round of the game.
        Input:
            - verbose, True/False to print information for the round.
        '''
        attendances = list()
        # Ask each agent to make a decision
        for i, agent in enumerate(self.agents):
            # Make decision
            decision = agent.make_decision()
            # Add to list of attendances
            attendances.append(decision)
        # Compute attendance and scores
        attendance, scores = self.environment.step(attendances)
        if verbose:
            print(f'\tAttendance = {attendance}\n')
        for i, agent in enumerate(self.agents):
            score = scores[i]
            # Learning rule is applied
            agent.update(score, attendances)
            if verbose:
                decision = attendances[i]
                decision_ = 'go' if decision == 1 else 'no go'
                print(f'\t\tAgent_{i} => {decision_}')
                agent.print_agent()
                print('')

    def run(self, verbose:bool=False):
        '''
        Runs the trial for the specified number of rounds.
        Input:
            - verbose, True/False to print information for the round.
        '''
        # Run the given number of rounds
        for round in range(self.num_rounds):
            if verbose:
                print('\n' + '-'*10 + f'Round {round}' + '-'*10 + '\n')
            self.play_round(verbose=verbose)

    def to_pandas(self) -> pd.DataFrame:
        '''
        Creates a pandas dataframe with the information from the current objects.
        Output:
            - pandas dataframe with the following six variables:
            
            Variables:
                * id_sim: a unique identifier for the simulation
                * threshold: the bar's threshold
                * round: the round number
                * attendance: the round's attendance
                * id_player: the player's number
                * decision: the player's decision
                * score: the player's score
                * model: the model's name
                * convergence: the maximum difference between 
                            two previous approximations of 
                            probability estimates
        '''
        data = {}
        data["id_sim"]= list()
        data["round"]= list()
        data["attendance"]= list()
        data["id_player"]= list()
        data["decision"]= list()
        data["score"]= list()
        if hasattr(self.agents[0], 'convergence'):
            include_convergence = True
            data["convergence"]= list()
        else:
            include_convergence = False
        for r in range(self.num_rounds):
            for i, a in enumerate(self.agents):
                data["id_sim"].append(self.id)
                data["round"].append(r)
                data["attendance"].append(self.environment.history[r])
                data["id_player"].append(a.number)
                data["decision"].append(a.decisions[r])
                data["score"].append(a.scores[r])
                if include_convergence:
                    data["convergence"].append(a.convergence[r])
        df = pd.DataFrame.from_dict(data)		
        df["model"] = self.model
        df["threshold"] = self.environment.threshold
        df["num_agents"] = self.environment.num_agents
        return df
    
    def simulate(self, num_episodes:int=1, file:str=None, verbose:bool=False):
        '''
        Runs a certain number of episodes.
        Input:
            - num_episodes, int with the number of episodes.
            - file, string with the name of file to save the data on.
            - verbose, True/False to print information.
        Output:
            - Pandas dataframe with the following variables:
                Variables:
                    * id_sim: a unique identifier for the simulation
                    * round: the round number
                    * attendance: the round's attendance
                    * id_player: the player's number
                    * decision: the player's decision
                    * score: the player's score
                    * model: the model's name
        '''		
        data_frames= list()
        # Run the number of episodes
        for t in tqdm(range(num_episodes), leave=False):
            self.id = uuid.uuid1()			
            for agent in self.agents:
                agent.reset()
            if verbose:
                print('\n' + '='*10 + f'Episode {t}' + '='*10 + '\n')
            # Reset environment for new episode
            self.reset()
            # Run the episode
            self.run(verbose=verbose)
            data_frames.append(self.to_pandas())
        data = pd.concat(data_frames, ignore_index=True)
        if file is not None:
            data.to_csv(file, index=False)
        return data

    def renderize(self, file:str=None, max_rounds:int=15):
        '''
        Plots the per round history as a grid.
        Input:
            - file, string with the name of file to save the data on.
        '''
        for round in range(max_rounds):
            self.play_round(verbose=0)				
            clear_output(wait=True)
            self.environment.render(file=file)
            sleep(self.sleep_time)

    def reset(self) -> None:
        self.environment.reset()
        for agent in self.agents:
            agent.reset()


class Experiment :
    '''
    Compares given models on a number of measures.
    '''

    def __init__(
                self, 
                agent_class: Agente, 
                fixed_parameters: Dict[str, any],
                free_parameters: Dict[str, any],
                simulation_parameters: Dict[str, any], 
                measures: Optional[List[str]]=[], 
                verbose: Optional[bool]=False,
        ) -> None:
        '''
        Input:
            - agent_class, class to create the agents.
            - fixed_parameters, a dictionary with the 
                fixed parameters of the class
            - free_parameters, a dictionary with the
                free parameters of the class
            - simulation_parameters, a diccionary with
                the number of rounds and episodes
            - measures, list of measures
            - verbose, True/False to print information.
        '''
        self.agent_class = agent_class
        self.fixed_parameters = fixed_parameters
        self.free_parameters = free_parameters
        bar, agents = self.initialize()
        self.environment = bar
        self.agents = agents
        self.num_rounds = simulation_parameters['num_rounds']
        self.num_episodes = simulation_parameters['num_episodes']
        self.measures = measures
        self.verbose = verbose
        self.data = None

    def initialize(
            self,
            num_agents: Optional[Union[int, None]]=None,
            agent_class: Optional[Union[int, None]]=None
        ) -> Tuple[Bar, List[Agente]]:
        if num_agents is None:
            num_agents = self.fixed_parameters['num_agents']
        if agent_class is None:
            agent_class = self.agent_class
            free_parameters = self.free_parameters.copy()
        else:
            free_parameters = self.free_parameters[agent_class.name()]
        bar = Bar(
            num_agents=num_agents,
            threshold=self.fixed_parameters['threshold']
        )
        fixed_parameters = self.fixed_parameters.copy()
        fixed_parameters['num_agents'] = num_agents
        agents = [
            agent_class(
                free_parameters=free_parameters, 
                fixed_parameters=fixed_parameters, 
                n=n
            ) for n in range(num_agents)
        ]
        return bar, agents

    def run_sweep1(
                self, \
                parameter: str, \
                values: List[float], \
                file_data: Optional[str]=None,
                ) -> None:
        '''
        Runs a parameter sweep of one parameter, 
        obtains the data and shows the plots on the given measures.
        Input:
            - parameter, a string with the name of the parameter.
            - values, a list with the parameter's values.
            - file, string with the name of file to save the plot on.
            - kwargs: dict with additional setup values for plots
        '''
        # Creates list of dataframes
        df_list= list()
        # Iterate over parameter values
        for value in tqdm(values, desc=f'Running models for each {parameter}'):
            if parameter == 'num_agents':
                bar, agents = self.initialize(num_agents=value)
                self.environment = bar
                self.agents = agents
            elif parameter == 'agent_class':
                bar, agents = self.initialize(agent_class=value)
                self.environment = bar
                self.agents = agents
                name = value.name()
            # Check if parameter modifies environment
            elif parameter == 'threshold':
                self.environment.threshold = value
            # Creates list for containing the modified agents
            if parameter not in ['agent_class']:
                # Iterate over agents
                for agent_ in self.agents:
                    # Modify agent's parameter with value
                    free_parameters_ = self.free_parameters.copy()
                    free_parameters_[parameter] = value
                    agent_.ingest_parameters(
                        fixed_parameters=self.fixed_parameters, 
                        free_parameters=free_parameters_
                    )
            # Create name
            name = f'{parameter}={value}'
            # Create simulation
            episode = Episode(
                environment=self.environment,\
                agents=self.agents,\
                model=name,\
                num_rounds=self.num_rounds
            )
            # Run simulation
            df = episode.simulate(
                num_episodes=self.num_episodes, 
                verbose=0
            )
            # Append dataframe
            df_list.append(df)
        # Concatenate dataframes
        self.data = pd.concat(df_list, ignore_index=True)
        if file_data is not None:
            self.data.to_csv(file_data, index=False)
            print(f'Data saved to {file_data}')
        print(f'Experiment with {parameter} finished')

    def run_sweep2(self, \
                    parameter1:str, \
                    values1:list,\
                    parameter2:str, \
                    values2:list, \
                    file_data:str=None
                        ):
        '''
        Runs a parameter sweep of one parameter, 
        obtains the data and shows the plots on the given measures.
        Input:
            - parameter1, a string with the name of the first parameter.
            - values1, a list with the first parameter's values.
            - parameter2, a string with the name of the second parameter.
            - values2, a list with the second parameter's values.
            - file_data, string with the name of file to save the plot on.
        '''
        # Creates list of dataframes
        df_list= list()
        # Creates list of agents
        for value1 in tqdm(values1):
            for value2 in tqdm(values2, leave=False):
                agents_parameter= list()
                for agent in self.agents:
                    agent_ = deepcopy(agent)
                    instruction = f'agent_.{parameter1} = {value1}'
                    exec(instruction)
                    instruction = f'agent_.{parameter2} = {value2}'
                    exec(instruction)
                    agents_parameter.append(agent_)
                # Creates name
                name = f'{parameter1}={value1}, {parameter2}={value2}'
                # Create simulation
                episode = Episode(environment=self.environment,\
                            agents=agents_parameter,\
                            model=name,\
                            num_rounds=self.num_rounds)
                # Run simulation
                df = episode.simulate(num_episodes=self.num_episodes, verbose=False)
                df[parameter1] = value1
                df[parameter2] = value2
                # Append dataframe
                df_list.append(df)
        # Concatenate dataframes
        self.data = pd.concat(df_list, ignore_index=True)
        if file_data is not None:
            self.data.to_csv(file_data, index=False)
            print(f'Data saved to {file_data}')
        print(f'Experiment with {parameter1} and {parameter2} finished')
    
