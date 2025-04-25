DASH_LINE = '-'*60

from Classes.agentes import Agente

def test_bar_is_full(agent:Agente, num_rounds=10) -> None:
    print('')
    print(DASH_LINE)
    print('Test bar is full')
    print(DASH_LINE)    
    action = 1
    other_player_actions = [1] * num_rounds
    state = [action, other_player_actions[0]]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = tuple(state)
    rollout(agent, other_player_actions, num_rounds)

def test_bar_has_capacity(agent:Agente, num_rounds=10) -> None:
    print('')
    print(DASH_LINE)
    print('Test bar has capacity')
    print(DASH_LINE) 
    agent.debug = True
    action = 0
    other_player_actions = [0] * num_rounds
    state = [action, other_player_actions[0]]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = tuple(state)
    rollout(agent, other_player_actions, num_rounds)

def test_alternation(agent:Agente, num_rounds=10) -> None:
    print('')
    print(DASH_LINE)
    print('Test other player alternates')
    print(DASH_LINE) 
    agent.debug = True
    action = 0
    other_player_actions = [0, 1] * (num_rounds // 2)
    state = [action, other_player_actions[0]]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = tuple(state)
    rollout(agent, other_player_actions, num_rounds)

def rollout(agent:Agente, other_player_actions:int, num_rounds:int) -> None:
    state = agent.prev_state_
    for i in range(num_rounds):
        print(f'---------- Round {i} ----------')
        preferences = agent.determine_action_preferences()
        print(f'Action preferences in state {state}: {preferences}')
        action = agent.make_decision()
        print('Chosen action:', action)
        other_player_action = other_player_actions[i]
        new_state = [action, other_player_action]
        print('State arrived:', new_state)
        payoff = agent.payoff(action, new_state)
        print(f'Payoff action {action}: {payoff}')
        agent.update(payoff, new_state)
        state = new_state
