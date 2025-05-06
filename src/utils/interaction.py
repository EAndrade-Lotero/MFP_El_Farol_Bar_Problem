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

    def __init__(self, environment:Bar, agents:List[any], model:str, num_rounds:int):
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
