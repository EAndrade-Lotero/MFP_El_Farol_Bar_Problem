import numpy as np
from random import randint, uniform

from typing import Optional, Union, Dict, List, Tuple

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
		#----------------------
		# Dealing with softmax overflow
		#----------------------
		self.fix_overflow = fix_overflow

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
			probabilities = self.softmax(preferences)
			if self.debug:
				print('Action probabilities:')
				print(f'no go:{probabilities[0]} ---- go:{probabilities[1]}')
			return probabilities[1]
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
		
	def softmax(
				self,
				preferences: List[float]
			) -> List[float]:
		'''
		Determines the softmax of the vector of preferences.
		Input:
			- preferences, list of preferences for actions
		Output:
			- List with softmax preferences
		'''
		# Broadcast inverse temperature and exponential
		# print('\tPreferences:', preferences)
		# numerator = np.exp(self.inverse_temperature * np.array(preferences))
		numerator = np.exp(self.inverse_temperature * np.array(preferences) - np.max(preferences)) # <= subtracted max for numerical stability
		num_inf = [np.isinf(x) for x in numerator]
		if sum(num_inf) == 1:
			softmax_values = [1 if np.isinf(x) else 0 for x in numerator]
			return softmax_values
		elif sum(num_inf) > 1:
			if self.fix_overflow:
				if self.debug:
					print(f'Overflow warning: {num_inf}')
				numerator = np.array([1 if np.isinf(x) else 0 for x in numerator])
			else:
				raise Exception(f'Error: softmax gives rise to numerical overflow (numerator: {numerator})!')
		# print('\tNumerator:', numerator)
		# find sum of values
		denominator = sum(numerator)
		assert(not np.isinf(denominator))
		if np.isclose(denominator, 0):
			if self.fix_overflow:
				if self.debug:
					print('Underflow warning:')
				softmax_values = [1 / len(numerator) for _ in numerator]
				return softmax_values
			else:
				raise Exception(f'Error: softmax gives rise to numerical overflow (denominator: {denominator})!')
		# print('\tDenominator:', denominator)
		# Return softmax using broadcast
		softmax_values = numerator / denominator
		assert(np.all([not np.isnan(n) for n in softmax_values])), f'numerator:{numerator} --- denominator: {denominator} --- preferences:{preferences}'
		return softmax_values

	def determine_action_preferences(self) -> List[float]:
		# To be defined by subclass
		pass

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

	def print_agent(self, ronda:int=None) -> str:
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
		self.inverse_temperature = free_parameters["inverse_temperature"]

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
		return 'CogMod'

	@staticmethod
	def bounds(fixed_parameters: Dict[str, any]) -> Dict[str, Tuple[int, int]]:
		return {
			'inverse_temperature': (1, 64),
		}
