import numpy as np

from itertools import product
from random import randint, uniform
from typing import Optional, Union, Dict, List, Tuple

from utils.agent_utils import ProxyDict, TransitionsFrequencyMatrix

class Agente () :
    '''
    Basic class for agents
    '''

    def __init__(
                self, 
                free_parameters: Optional[Dict[str,any]]={}, 
                fixed_parameters: Optional[Dict[str,any]]={}, 
                n: Optional[int]=1,
                fix_overflow: Optional[bool]=True
            ) -> None:
        #----------------------
        # Initialize lists
        #----------------------
        self.decisions = []
        self.scores = []
        self.prev_state_ = None
        self.debug = False
        self.number = n
        #----------------------
        # Parameter bookkeeping
        #----------------------
        self.ingest_parameters(fixed_parameters, free_parameters)

    def make_decision(self) -> int:
        '''
        Agent decides whether to go to the bar or not.
        Output:
            - A decision 0 or 1
        '''
        # Agent recalls previous state?
        if self.prev_state_ is not None:
            go_prob = self.go_probability()
            probabilities = [1 - go_prob, go_prob]
            decision = np.random.choice(
                a=[0,1],
                size=1,
                p=probabilities
            )[0]
            if isinstance(decision, np.int64):
                decision = int(decision)
            return decision
        else:
            # no previous data, so make random decision
            return randint(0, 1)
        
    def go_probability(self) -> float:
        '''
        Agent returns the probability of going to the bar
        according to its model.
        Output:
            - p, float representing the probability that the
                agent decides to go to the bar.
        '''
        # Agent recalls previous state?
        if self.prev_state_ is not None:
            # determine action preferences given previous state
            preferences = self.determine_action_preferences()
            if self.debug:
                print('Action probabilities:')
                print(f'no go:{preferences[0]} ---- go:{preferences[1]}')
            return preferences[1]
        else:
            # no previous data
            return 0.5
        
    def payoff(
                self, 
                action: int, 
                state: List[int]
            ) -> int:
        '''
        Determines the payoff of an action given the bar's attendance.
        Input:
            - action, go = 1 or no_go = 0
            - state, list of decisions of all agents
        Output:
            - List with payoff for the action
        '''
        attendance = sum([x == 1 or x == '1' for x in state])
        if action == 0 or action == '0':
            return 0
        elif attendance <= self.threshold * self.num_agents:
            return 1
        else:
            return -1
        
    def update(self, score:int, obs_state:List[int]) -> None:
        '''
        Agent updates its model.
        Input:
            - score, a number 0 or 1.
            - obs_state_, a tuple with the sate of current round,
                        where each argument is 0 or 1.
        '''
        # Update records
        self.scores.append(score)
        action = obs_state[self.number]
        try:
            action = int(action)
        except Exception as e:
            print(f'Error: action of type {type(action)}. Type int was expected. (previous actions: {self.decisions})')
            raise Exception(e)
        self.decisions.append(action)
        self.prev_state_ = tuple(obs_state)

    def reset(self) -> None:
        '''
        Restarts the agent's data for a new trial.
        '''
        self.decisions = []
        self.scores = []
        self.prev_state_ = None

    def restart(self) -> None:
        # To be defined by subclass
        pass

    def ingest_parameters(
                self, 
                fixed_parameters:Dict[str,any], 
                free_parameters:Dict[str,any]
            ) -> None:
        '''
        Ingests parameters from the model.
        Input:
            - fixed_parameters, dictionary with fixed parameters
            - free_parameters, dictionary with free parameters
        '''
        self.fixed_parameters = fixed_parameters
        self.free_parameters = free_parameters
        self.threshold = fixed_parameters["threshold"]
        self.num_agents = int(fixed_parameters["num_agents"])

    def __str__(self) -> str:
            '''
            Returns a string with the state of the agent on a given round.
            Input:
                - ronda, integer with the number of the round.
            Output:
                - string with a representation of the agent at given round.
            '''
            try:
                ronda = len(self.decisions) - 1
            except:
                ronda = 0
            try:
                decision = self.decisions[ronda]
            except:
                decision = "nan"
            try:
                score = self.scores[ronda]
            except:
                score = "nan"
            return f"No.agent:{self.number}\nDecision:{decision}, Score:{score}"

    @staticmethod
    def name():
        return 'Agent'



class MFP(Agente) :
    '''
    Implements an agent using the Markov Fictitious Play learning rule 
    for multiple players.
    This model conditions G on the previous actions vector, the full-state.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str,any]]={}, 
                fixed_parameters:Optional[Dict[str,any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        #--------------------------------------------
        # Create counters
        #--------------------------------------------
        self.states = [0]
        self.reset()
        self.belief_strength = free_parameters['belief_strength']
        assert(self.belief_strength > 0)
        self.states = list(product([0, 1], repeat=self.num_agents))
        self.reset()

    def determine_action_preferences(self) -> List[float]:
        '''
        Agent determines their preferences to go to the bar or not.
        Input:
            - state, list of decisions of all agents
        Output:
            - List with no go preference followed by go preference
        '''
        if self.prev_state_ is not None:
            eus = [self.exp_util(action) for action in [0,1]]
            prefs = [0, 0]
            prefs[1] = 1 if eus[1] > eus[0] else 0
            prefs[0] = 1 if eus[0] > eus[1] else 0
        else:
            prefs = [0.5, 0.5]
        if self.debug:
            print('Expected utilities:')
            print(f'no go:{eus[0]} ---- go:{eus[1]}')
            print(f'Action preferences:')
            print(f'no go:{prefs[0]} ---- go:{prefs[1]}')
        return prefs
    
    def exp_util(self, action:int) -> float:
        '''
        Evaluates the expected utility of an action.
        Input:
            - prev_state, a tuple with the state of the previous round, 
                        where each argument is 0 or 1.
            - action, which is a possible decision 0 or 1.
        Output:
            - The expected utility (float).
        '''
        if action == 0:
            return 0
        else:
            prev_sate = self.get_prev_state()
            numerator = self.count_bar_with_capacity(prev_sate) + self.belief_strength
            denominator = self.count_states(prev_sate) + len(self.states) * self.belief_strength
            prob_capacity = numerator / denominator
            prob_crowded = 1 - prob_capacity
            eu = prob_capacity - prob_crowded
            if self.debug:
                print(f'{prob_capacity=} --- {prob_crowded=}')
        return eu

    def update(self, score:int, obs_state:Tuple[int]):
        '''
        Agent updates its model using the observed frequencies.
        Input:
            - score, a number 0 or 1.
            - obs_state_, a tuple with the sate of current round,
                        where each argument is 0 or 1.
        Input:
        '''
        # Update records
        self.scores.append(score)
        self.decisions.append(obs_state[self.number])
        # Agent recalls previous state?
        if self.prev_state_ is not None:
            prev_state = self.get_prev_state()
            # Increment count of states
            self.count_states.increment(prev_state)
            # Find other player's attendance
            action = obs_state[self.number]
            other_players_attendance = sum(obs_state) - action
            if other_players_attendance < int(self.threshold * self.num_agents):
                # Increment count of bar with capacity given previous state
                self.count_bar_with_capacity.increment(prev_state)
        if self.debug:
            print(f'I see the previous state: {prev_state}')
            print('I recall the following frequencies of states:')
            print(self.count_states)
            print('I recall the following frequencies of bar with capacity:')
            print(self.count_bar_with_capacity)
        # Update previous state
        self.prev_state_ = obs_state

    def _get_error_message(
                self, 
                new_prob: float, 
                transition: Dict[any, any], 
                prev_state: Tuple[int]
            ) -> str:
        error_message = f'Error: Improper probability value {new_prob}.\n'
        error_message += f'Transition:{transition}\n'
        error_message += f'Transition counts:{self.count_transitions(transition)}\n'
        error_message += f'Prev. state counts:{self.count_states(tuple(prev_state))}'
        return error_message	

    def reset(self) :
        '''
        Restarts the agent's data for a new trial.
        '''
        super().reset()
        self.prev_state_ = None
        self.count_states = ProxyDict(keys=self.states, initial_val=0)
        self.count_bar_with_capacity = ProxyDict(
            keys=self.states,
            initial_val=0
        )

    def __str__(self, ronda:int=None) -> str:
        '''
        Returns a string with the state of the agent on a given round.
        Input:
            - ronda, integer with the number of the round.
        Output:
            - string with a representation of the agent at given round.
        '''
        if ronda is None:
            try:
                ronda = len(self.decisions) - 1
            except:
                ronda = 0
        try:
            decision = self.decisions[ronda]
        except:
            decision = "nan"
        try:
            score = self.scores[ronda]
        except:
            score = "nan"
        print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}")
        tp = TransitionsFrequencyMatrix(num_agents=self.num_agents)
        tp.from_proxydict(self.trans_probs)
        print(tp)

    def get_prev_state(self):
        return self.prev_state_

    @staticmethod
    def name():
        return 'MFP-M3'