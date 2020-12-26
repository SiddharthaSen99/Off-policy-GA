import gym
import numpy as np
import math

class GaTRPO():
    def __init__(self, buckets=(1, 1, 6, 12), num_episodes=1000, min_lr=0.1, min_epsilon=0.1, discount=1.0, decay=25):
        
        # Genetic Algorithm Variables
        self.number_of_target_policies = 100
        self.number_of_behavior_policies = 1

        # Environment Variables
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay

        self.env = gym.make('CartPole-v0')

        # [position, velocity, angle, angular velocity]
        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50) / 1.]

        self.Q_tables = [np.zeros(self.buckets + (self.env.action_space.n,)) for i in range(self.number_of_target_policies)]


    def discretize_state(self, obs):
        discretized = list()
        for i in range(len(obs)):
            scaling = (obs[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def choose_action(self, state, index):
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample() 
        else:
            return np.argmax(self.Q_tables[index][state])

    def update_q(self, state, action, reward, new_state,index):
        self.Q_tables[index][state][action] += self.learning_rate * (reward + self.discount * np.max(self.Q_tables[index][new_state]) - self.Q_tables[index][state][action])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def train(self,index):
        for e in range(self.num_episodes):
            current_state = self.discretize_state(self.env.reset())

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False

            while not done:
                action = self.choose_action(current_state,index)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                self.update_q(current_state, action, reward, new_state,0)
                current_state = new_state

        print('Finished training!')

    def run(self,index):
        self.env = gym.wrappers.Monitor(self.env,'cartpole',force = True)
        t = 0
        done = False
        current_state = self.discretize_state(self.env.reset())
        while not done:
                self.env.render()
                t = t+1
                action = self.choose_action(current_state,index)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                current_state = new_state
            
        return t   
            


if __name__ == "__main__":
    agents = GaTRPO()
    agents.train(0)
    t = agents.run(0)
    print("Time", t)
