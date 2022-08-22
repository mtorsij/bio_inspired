# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
from data_init import data_init
from matplotlib import pyplot as plt
import pandas as pd

# =============================================================================
# RACETRACK
# =============================================================================
# racetrack = np.array([[-1,0,0,0,2],
#                             [-1,0,0,0,2],
#                             [-1,0,0,-1,-1],
#                             [-1,0,0,-1,-1],
#                             [-1,1,1,-1,-1]])

# racetrack = np.array([[-1,-1,-1,-1,-1,0,0,0,0,2],
#                       [-1,0,0,0,0,0,0,0,0,2],
#                       [-1,0,0,0,0,0,0,0,0,2],
#                       [-1,0,0,0,0,0,0,0,0,2],
#                       [-1,0,0,0,-1,-1,0,0,-1, -1],
#                       [-1,0,0,0,-1,-1,0,0,-1,-1],
#                       [-1,0,0,0,-1,0,0,-1,-1,-1],
#                       [-1,0,0,0,0,0,0,-1,-1,-1],
#                       [-1,0,0,0,0,-1,-1,-1,-1,-1],
#                       [-1,1,1,1,1,-1,-1,-1,-1,-1]])

# racetrack = np.load('racetrack.npy')

# Get racetrack from excel file
racetrack_df = pd.read_excel('racetrack_2_roads.xlsx')
racetrack = racetrack_df.to_numpy()

# Speed limits on two track racetrack
fast_speed_limit = 6
slow_speed_limit = 3

rows_racetrack = len(racetrack)
cols_racetrack = len(racetrack[0])

# Racetrack finish and start
finish_line = np.array([np.array([i,cols_racetrack-1]) for i in range(rows_racetrack) if racetrack[i,cols_racetrack-1] == 2])
start_line = np.array([np.array([rows_racetrack-1,j]) for j in range(cols_racetrack) if racetrack[rows_racetrack-1,j] == 1])

def finish_crossed_check(state, action):
    new_state = get_new_state(state, action)
    old_cell, new_cell = state[0:2], new_state[0:2]
    
    rows = np.array(range(new_cell[0],old_cell[0]+1))
    cols = np.array(range(old_cell[1],new_cell[1]+1))
    fin = set([tuple(x) for x in finish_line])
    row_col_matrix = [(x,y) for x in rows for y in cols]
    intersect = [x for x in row_col_matrix if x in fin]
    
    return len(intersect) > 0
    
def out_of_bounds_check(state, action):
    new_state = get_new_state(state, action)
    
    if new_state[0] < 0 or new_state[0] >= rows_racetrack or new_state[1] < 0 or new_state[1] >= cols_racetrack:
        return True
    
    else:
        return racetrack[tuple(new_state[0:2])] == -1 

# =============================================================================
# DATA 
# =============================================================================

# Initialize matrices (Q random values, C and pi zeros)
Q_values,C_values,pi = data_init(cols_racetrack, rows_racetrack, fast_speed_limit)
rewards = []

# Learning parameters
epsilon = 0.1
gamma = 1

# Construct pi based on racetrack
for i in range(rows_racetrack):
    for j in range(cols_racetrack):
        if racetrack[i,j]!=-1:                                            
            for k in range(fast_speed_limit):
                for l in range(fast_speed_limit):
                    pi[i,j,k,l] = np.argmax(Q_values[i,j,k,l])                     

# =============================================================================
# RACECAR
# =============================================================================
def get_available_actions(state):
    all_actions = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]
    all_actions = [np.array(x) for x in all_actions]

    available_actions = []
    
    # Road with slow speed limit
    if racetrack[tuple(state[0:2])] == 3:
        
            # If speed in y direction greater than 3 when coming onto the road always -1
            if state[2] >= slow_speed_limit and state[3] < slow_speed_limit:
                # Adjust all actions list accordingly, 1D
                adjusted_actions = [0,1,3]
                
                # Loop over all actions
                for i in adjusted_actions:
                    new_vel = np.add(state[2:4],all_actions[i])
                    if new_vel[1] < slow_speed_limit and new_vel[1] >= 0 and ~(new_vel[0] == 0 and new_vel[1] == 0):
                        available_actions.append(i)
                        
            # If speed in x direction greater than 3 when coming onto the road always -1
            elif state[3] >= slow_speed_limit and state[2] < slow_speed_limit:
                # Adjust all actions list accordingly, 1D
                adjusted_actions = [0,1,2]
                
                # Loop over all actions
                for i in adjusted_actions:
                    new_vel = np.add(state[2:4],all_actions[i])
                    if new_vel[0] < slow_speed_limit and new_vel[0] >= 0 and ~(new_vel[0] == 0 and new_vel[1] == 0):
                        available_actions.append(i)
            
            # If speed components of both directions are too fast, decrease both
            elif state[3] >= slow_speed_limit and state[2] >= slow_speed_limit:
                available_actions.append(0)
            
            # Speed in both directions is good
            else:    
                for i,x in zip(range(9),all_actions):
                    new_vel = np.add(state[2:4],x)
                    if (new_vel[0] < slow_speed_limit) and (new_vel[0] >= 0) and (new_vel[1] < slow_speed_limit) and (new_vel[1] >= 0) and ~(new_vel[0] == 0 and new_vel[1] == 0):
                        available_actions.append(i)
    
    # Normal road with fast speed limit
    else:    
        for i,x in zip(range(9),all_actions):
            new_vel = np.add(state[2:4],x)
            if (new_vel[0] < fast_speed_limit) and (new_vel[0] >= 0) and (new_vel[1] < fast_speed_limit) and (new_vel[1] >= 0) and ~(new_vel[0] == 0 and new_vel[1] == 0):
                available_actions.append(i)
    
    
    available_actions = np.array(available_actions)
            
    return available_actions

def back_to_start():
    state = np.zeros(4, dtype='int')
    state[0] = rows_racetrack-1
    state[1] = np.random.choice(start_line[:,1])

    return state

def map_to_1D(action):
    all_actions = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]
    for i,x in zip(range(9),all_actions):
        if action[0]==x[0] and action[1]==x[1]:
            return i
    
def map_to_2D(action):
    all_actions = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]
    return all_actions[action]

# =============================================================================
# ENVIRONMENT
# =============================================================================
# Get new state 
def get_new_state(state, action):
    # Make a copy of current state
    new_state = state.copy()
    
    # Before the accelerate or decelerate action is applied, the car continues
    # to move with current velocity
    new_state[0] = state[0] - state[2]
    new_state[1] = state[1] + state[3]
    new_state[2] = state[2] + action[0]
    new_state[3] = state[3] + action[1]
    
    return new_state

# Perform step
def step(state, action, count_steps):
    episode['Action'].append(action)
    reward = -1
    
    if finish_crossed_check(state,action):
        new_state = get_new_state(state, action)
        
        episode['Reward'].append(reward)
        episode['State'].append(new_state)
        count_steps += 1
    
        return None, new_state, step_count
        
    elif out_of_bounds_check(state,action):
        new_state = back_to_start()
    else:
        new_state = get_new_state(state, action)
    
    episode['Reward'].append(reward)
    episode['State'].append(new_state)  
    count_steps += 1
    
    return reward, new_state, count_steps
    
# =============================================================================
# MONTE CARLO CONTROL
# =============================================================================
# Get behavourial policy action
def get_behavioural_policy_action(state, possible_actions,pi_policy):
    # Select random or optimal policy action
    if np.random.rand() > epsilon and pi_policy[tuple(state)] in possible_actions:
        action = pi_policy[tuple(state)]
    else:
        action = np.random.choice(possible_actions)
    
    # Calculate and append probability to episode
    get_probability_behaviour(state, action, possible_actions)

    return action

# Get target policy action
def get_target_policy_action(state, possible_actions,pi_policy):
     # Select optimal policy action or random if not in possible actions
     if pi_policy[tuple(state)] in possible_actions:
         action = pi_policy[tuple(state)]
     else:
         action = np.random.choice(possible_actions)
    
     return action

# Get probability behaviour
def get_probability_behaviour(state, action, possible_actions):
    best_action = pi[tuple(state)]
    num_actions = len(possible_actions)
    
    if best_action in possible_actions:
        if action == best_action:
            prob = 1 - epsilon + epsilon/num_actions
        else:
            prob = epsilon/num_actions
    else:
        prob = 1/num_actions
    
    episode['Prob'].append(prob)
            
# =============================================================================
# SIMULATION
# =============================================================================
steps = 10000

for i in range(steps):
    
    # MONTE CARLO CONTROL
    # Start new episode
    episode = {'State':[], 'Action':[], 'Prob':[], 'Reward':[None]}
    step_count = 0
    state = back_to_start()
    episode['State'].append(state)
    
    # Generate episode
    reward = -1
    while reward!=None:
        possible_actions = get_available_actions(state)
        action = map_to_2D(get_behavioural_policy_action(state, possible_actions,pi))
        reward, state, step_count = step(state, action, step_count)
    
    G = 0
    W = 1
    T = len(episode['Action'])
    
    for t in range(T-1,-1,-1):
        G = gamma * G + episode['Reward'][t+1]
        S_t = tuple(episode['State'][t])
        A_t = map_to_1D(episode['Action'][t])
        
        S_list = list(S_t)
        S_list.append(A_t)
        SA = tuple(S_list)
        
        C_values[SA] += W
        Q_values[SA] += (W*(G-Q_values[SA]))/(C_values[SA])            
        pi[S_t] = np.argmax(Q_values[S_t])
        if A_t != pi[S_t]:
            break
        W /= episode['Prob'][t]
    
    # EVALUATE POLICY
    if i%10 == 9:
        print('evaluating ' + str(i))
        # Start new episode
        episode = {'State':[], 'Action':[], 'Prob':[], 'Reward':[None]}
        state = back_to_start()
        step_count = 0
        episode['State'].append(state)
        
        # Generate episode
        reward = -1
        while reward!=None:
            possible_actions = get_available_actions(state)
            action = map_to_2D(get_target_policy_action(state, possible_actions,pi))
            reward, state, step_count = step(state, action, step_count)
            
        # Append result to rewards list
        rewards.append(sum(episode['Reward'][1:]))

# =============================================================================
# PLOT REWARDS
# =============================================================================

ax, fig = plt.subplots(figsize=(30,15))
x = np.arange(1,len(rewards)+1)
plt.plot(x*10, rewards, linewidth=0.8, color = 'red')
plt.xlabel('Episode number', size = 40)
plt.ylabel('Reward',size = 40)
plt.title('Plot of Reward vs Episode Number',size=40)
plt.xticks(size=40)
plt.yticks(size=40)

# =============================================================================
# PLOT ROUTE
# =============================================================================
# Start new episode
episode = {'State':[], 'Action':[], 'Prob':[], 'Reward':[None]}
states = []
state = back_to_start()
step_count = 0
episode['State'].append(state)
states.append(state[0:2])

# Generate episode
reward = -1
while reward!=None:
    possible_actions = get_available_actions(state)
    action = map_to_2D(get_target_policy_action(state, possible_actions,pi))
    reward, state, step_count = step(state, action, step_count)
    states.append(state[0:2])

states.pop()
    
for state in states:
    racetrack[tuple(state)] = 10

plt.figure()
plt.imshow(racetrack)


