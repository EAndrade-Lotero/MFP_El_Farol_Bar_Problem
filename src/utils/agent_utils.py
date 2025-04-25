import torch
import numpy as np
from pathlib import Path
from random import choice
from copy import deepcopy
from itertools import product
from prettytable import PrettyTable
from torch.nn import MSELoss # CrossEntropyLoss
from typing import List, Tuple, Dict, Union, Optional


class ProxyDict :

    def __init__(
                self,
                keys: List[Tuple[int]],
                initial_val: float,
                round_dec: Optional[int]=2
            ) -> None:
        assert(isinstance(keys, list))
        self._keys = keys
        self.initial_val = initial_val
        self.data_dict = {key:self.initial_val for key in self._keys}
        self.round_dec = round_dec

    def __len__(self) -> int:
        return len(self.data_dict)

    def reset(self):
        self.data_dict = {key:self.initial_val for key in self._keys}

    def __call__(self, key: List[int]) -> float:
        if isinstance(key, List):
            key = tuple(key)
        return self.data_dict[key]

    def update(self, key: List[int], new_value: any) -> None:
        if isinstance(key, List):
            key = tuple(key)
        self.data_dict[key] = new_value

    def increment(self, key: List[int]) -> None:
        if isinstance(key, List):
            key = tuple(key)
        self.data_dict[key] += 1

    def keys(self) -> List[Tuple[int]]:
        return self._keys

    def sum(self) -> float:
        return sum(self.data_dict.values())
    
    def normalize(self) -> None:
        row_sum = self.sum()
        row_sum = row_sum if row_sum != 0 else 1
        for key in self.data_dict.keys():
            if isinstance(key, List):
                key = tuple(key)
            self.data_dict[key] /= row_sum

    def as_array(self) -> np.ndarray:
        return np.array(list(self.data_dict.values()))

    def from_dict(self, given_dict: Dict[Tuple[int], float]) -> None:
        for key, value in given_dict.items():
            if isinstance(key, List):
                key = tuple(key)
            self.update(
                key=key,
                new_value=value
            )

    def __str__(self) -> str:
        table = PrettyTable(field_names=list(self.data_dict.keys()))
        row = [round(value, self.round_dec) for value in self.data_dict.values()]
        table.add_row(row)
        return str(table)


class TransitionsFrequencyMatrix :

    def __init__(
                self,
                num_agents: int,
                round_dec: Optional[int]=2,
                uniform: Optional[bool]= True
            ) -> None:
        self.num_agents = num_agents
        self.round_dec = round_dec
        num_rows = np.power(2, self.num_agents)
        num_cols = np.power(2, self.num_agents)
        if uniform:
            self.trans_freqs = np.ones((num_rows, num_cols)) * (1 / num_cols)
        else:
            self.trans_freqs = np.zeros((num_rows, num_cols))

    def reset(self):
        num_rows = np.power(2, self.num_agents)
        num_cols = np.power(2, self.num_agents)
        self.trans_freqs = np.ones((num_rows, num_cols)) * (1 / num_cols)

    def __len__(self) -> int:
        return self.trans_freqs.shape[0]

    def __call__(self, transition: Tuple[List[int], List[int]]) -> float:
        row, col = self.get_indices(transition)
        return self.trans_freqs[row, col]

    def get_indices(self, transition: Tuple[List[int], List[int]]) -> Tuple[int, int]:
        prev_state, state = transition
        row = int("".join(str(x) for x in prev_state), 2)
        col = int("".join(str(x) for x in state), 2)
        return (row, col)

    def get_state_from_index(self, index: int) -> Tuple[int]:
        binary = "{0:b}".format(index)
        binary = list(binary)
        binary = [0 for _ in range(self.num_agents - len(binary))] + binary
        return tuple(binary)

    def get_convergence(self):
        return np.max(self.trans_freqs)
    
    def update(
                self, 
                transition: Tuple[List[int], List[int]],
                value: float
            ) -> None:
        row, col = self.get_indices(transition)
        self.trans_freqs[row, col] = value

    def increment(
                self, 
                transition: Tuple[List[int], List[int]],
            ) -> None:
        row, col = self.get_indices(transition)
        self.trans_freqs[row, col] += 1

    def normalize(self, rows: Optional[bool]=True) -> None:
        if rows:
            for i in range(self.trans_freqs.shape[0]):
                row_sum = sum(self.trans_freqs[i])
                row_sum = row_sum if row_sum != 0 else 1
                self.trans_freqs[i] /= row_sum
        else:
            for i in range(self.trans_freqs.shape[1]):
                col_sum = sum(self.trans_freqs[i])
                col_sum = col_sum if col_sum != 0 else 1
                self.trans_freqs[:,i] /= col_sum

    def from_dict(self, trans_probs: Dict[Tuple[int], float]) -> None:
        for transition in trans_probs.keys():
            self.update(
                transition=transition,
                value=trans_probs[transition]
            )

    def from_proxydict(self, trans_probs: Dict[Tuple[int], float]) -> None:
        for transition in trans_probs.keys():
            self.update(
                transition=transition,
                value=trans_probs(transition)
            )

    def __str__(self) -> str:
        states = list(product([0,1], repeat=self.num_agents))
        table = PrettyTable(field_names=[''] + states)
        for x in states:
            row = [x] + [round(self((x,y)), self.round_dec) for y in states]
            table.add_row(row)
        return str(table)

