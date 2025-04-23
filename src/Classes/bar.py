'''
Class with the El Farol bar environment
'''

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Bar :
    '''
    Class for playing El Farol bar problem. Keeps tally of the number of players
    attending the bar each round and returns a list of scores depending on whether
    the bar overcrouds or not.
    '''
    
    def __init__(self, num_agents, threshold) :
        self.num_agents = num_agents
        self.threshold = threshold
        self.history = []

    def step(self, decisions:list) -> list :
        '''
        Computes the scores on the basis of the attendance.
        Input:
            - decisions, list with each player's decision (1=GO, 0=NO GO)
        Output:
            - attendance, number of players attending the bar
            - scores, list with a score for each player according to the following payoff matrix:
                1, if agent goes and attendance <= threshold*num_agents
                -1, if agent goes and attendance > threshold*num_agents
                0, if agent does not go
        '''
        assert(all([a in [0,1] for a in decisions]))
        attendance = sum(decisions)
        self.history.append(decisions)
        scores = []
        for a in decisions:
            if a == 1:
                if attendance <= self.threshold * self.num_agents:
                    scores.append(1)
                else:
                    scores.append(-1)
            else:
                scores.append(0)
        return attendance, scores

    def reset(self):
        '''
        Goes back to initial state.
        '''
        self.history = []

    def render(self, file, num_rounds:int=15):
        '''
        Renders the history of attendances.
        '''
        # Use only last num_rounds rounds
        history = self.history[-num_rounds:]
        len_padding = num_rounds - len(history)
        if len_padding > 0:
            history = [[2 for _ in range(self.num_agents)] for i in range(len_padding)] + history
        # Convert the history into format player, round
        decisions = [[h[i] for h in history] for i in range(self.num_agents)]
        # Create plot
        fig, axes = plt.subplots(figsize=(0.5*num_rounds,2))
        # Determine step sizes
        step_x = 1/num_rounds
        step_y = 1/self.num_agents
        # Determine color
        go_color='blue'
        no_go_color='lightgray'
        # Draw rectangles (go_color if player goes, gray if player doesnt go)
        tangulos = []
        for r in range(num_rounds):
            for p in range(self.num_agents):
                if decisions[p][r] == 1:
                    color = go_color
                elif decisions[p][r] == 0:
                    color = no_go_color
                else:
                    color = 'none'
                # Draw filled rectangle
                tangulos.append(
                    patches.Rectangle(
                        (r*step_x,p*step_y),step_x,step_y,
                        facecolor=color
                    )
                )
        for r in range(len_padding, num_rounds + 1):
            # Draw border
            tangulos.append(
                patches.Rectangle(
                    (r*step_x,0),0,1,
                    edgecolor='black',
                    facecolor=no_go_color,
                    linewidth=1
                )
            )
        for p in range(self.num_agents + 1):
            # Draw border
            tangulos.append(
                patches.Rectangle(
                    (len_padding*step_x,p*step_y),1,0,
                    edgecolor='black',
                    facecolor=no_go_color,
                    linewidth=1
                )
            )
        for t in tangulos:
            axes.add_patch(t)
        axes.axis('off')
        plt.savefig(file, dpi=300)
        plt.close()
        