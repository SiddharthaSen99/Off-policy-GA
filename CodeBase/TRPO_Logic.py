# https://en.wikipedia.org/wiki/Inverted_pendulum
# environment: https://github.com/openai/gym/wiki/CartPole-v0
# 2 actions: 0:push_left, 1:push_right
# 4 observations: 0:cart_position ; 1:cart_volecity ; 2:pole_angle; 3:pole_volecity_at_tip

import gym
import numpy as np
import random
import math


##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
#########################START OF PARAMETERS######################################

environment_name = 'CartPole-v0'
environment = gym.make(environment_name)
environment.seed(42)


# 4 observations: 0:cart_position ; 1:cart_volecity ; 2:pole_angle; 3:pole_volecity_at_tip
# number of discrete states  per state dimension
number_states = (1, 1, 6, 3)  # (x, x', theta, theta')

# 2 actions: 0:push_left, 1:push_right
number_actions = environment.action_space.n # (left, right)

# bounds for each discrete state
state_bounds = list(zip(environment.observation_space.low, environment.observation_space.high))
state_bounds[1] = [-1, 1]
state_bounds[3] = [-math.radians(50), math.radians(50)]

# simulation related constants
Gene = 1000
generations = 50
number_of_polcies = 100
max_step = 250
success_to_end = 100
pretest_number = 199
K = 50

# learning related constants
min_explore_rate = 0.01
min_learning_rate = 0.1


###########################END OF PARAMETERS######################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################



# create qTable with zeros





def observation_to_state(observation):
    states_list = []
    for i in range(len(observation)):
        if observation[i] <= state_bounds[i][0]:
            state_index = 0
        elif observation[i] >= state_bounds[i][1]:
            state_index = number_states[i] - 1
        else:
            # map the state bounds to the state array
            bound_width = state_bounds[i][1] - state_bounds[i][0]
            offset = (number_states[i]-1)*state_bounds[i][0]/bound_width
            scaling = (number_states[i]-1)/bound_width
            state_index = int(round(scaling*observation[i] - offset))
        states_list.append(state_index)
    return tuple(states_list)






def select_action(policy,state, explore_rate):
    # select a random action
    if random.random() < explore_rate:
        action = environment.action_space.sample()
    # select the action with the highest q
    else:
        action = np.argmax(policy[state])
    return action





def get_explore_rate(T):
    return max(min_explore_rate, min(1, 1.0 - math.log10((T+1)/25)))





def get_learning_rate(T):
    return max(min_learning_rate, min(0.5, 1.0 - math.log10((T+1)/25)))







def get_policies(number_of_polcies):

    # Here we select number of polcies amount of polcies to begin with all
    # Initiailzed randomly and gaussian normalized 

    polcies = []

    for i in range(number_of_polcies):
        policy = np.random.randn(number_states + (number_actions,))
        policies.append(policy)

    return policies









def get_behaviour_policy(policies):

    # Get the required behaviour polcy using a suitable method
    behaviour_policy = np.zero(number_states + (number_actions,))
    # Currently planned: Minimal KL divergence sum policy.

    return behaviour_policy




def get_surrogate_loss(policy,behaviour_policy):
    loss = 0

    # Implementation of TRPO logic

    return loss

def select_k_best(K,surrogate_loss,policies):

    # selects the first K policies with lowest surrogate losss
    surrogate_loss.sort()
    
    next_gen = []
    for i in range(len(K)):
        next_gen.append(policies[surrogate_loss[i][1]])

    return next_gen

def mutate(policies):

    
    return policies


if __name__ == "__main__":

    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.98

    num_success = 0


    # Gets the intended list of Polcies and Behaviour Policy
    policies = get_policies(number_of_polcies)
    behaviour_policy = get_behaviour_policy(policies)
    




    for T in range(generations):
        
        surrogate_loss = []
        # reset the environment
        policy_count = 0
        for policy in range(policies):

            observation = environment.reset()
            loss = get_surrogate_loss(policy,behaviour_policy)
            surrogate_loss.append([loss,policy_count])
            

            if done:
                print('Iteration No: ',T + 1)
                # after pretest_number
                if (T >= pretest_number): num_success += 1
                else: num_success = 0
                break

            policy_count += 1

        surrogate_loss.sort()

        policies = select_k_best(K,surrogate_loss,policies)
        policies = mutate(policies)