import gym
import numpy as np
from gym import wrappers

# Plot the results
import plotly
import plotly.graph_objs as go

# Globals
n_generations = 0
plot_data = []
final_games = 10
score_requirement = 50
population_size = 100 
generation_limit = 50  # Max number of generations
steps_limit = 300 # Max number of steps in a game
sigma = 0.1  # Noise standard deviation
alpha = 0.0005  # Learning rate

RNG_SEED = 8
NULL_ACTION = 0

def create_plot():
	global plot_data
	global n_generations
	trace = go.Scatter(
		x = np.linspace(0,1, n_generations),
		y = plot_data,
		mode = 'lines+markers',
		fill='tozeroy'
	)
	data = [trace]
	plotly.offline.plot({"data": data, "layout": go.Layout(title = "LunarLander")}, filename = "LunarLander_plot")
	
def genetic_algorithm():
	env = gym.make("LunarLander-v2")
	env.reset()
	input_size = env.observation_space.shape[0]
	output_size = env.action_space.n
	
	env.seed(RNG_SEED)
	np.random.seed(RNG_SEED)
	
	global n_generations
	# Initial weights
	W = np.zeros((input_size, output_size))

	for gen in range(generation_limit):
		# Keep track of Returns
		R = np.zeros(population_size)
		
		# Generate noise
		N = np.random.randn(population_size, input_size, output_size)
		# Try every set of new values and keep track of the returns
		for j in range(population_size):
			W_ = W + sigma * N[j]
			R[j] = run_episode(env, W_, False)

	    # Update weights on the basis of the previous runned episodes
	    # Summation of episode_weight * episode_reward
		weighted_weights = np.matmul(N.T, R).T
		new_W = W + alpha / (population_size * sigma) * weighted_weights
		W = new_W
		
		gen_mean = np.mean(R)

		plot_data.append(gen_mean)
		n_generations += 1
		
		print("Generation {}, Population Mean: {}".format(gen, gen_mean))
		if gen_mean >= score_requirement:
			break
	
	print("Running final games")
	for i in range(final_games):
		print("episode {}, score: {}".format(i, run_episode(env, W, True)))
	return

def run_episode(env, weight, render = False):
	obs = env.reset()
	episode_reward = 0
	done = False
	step = 0
	while not done:
		if(render):
			env.render()
		if(step > steps_limit):
			move = NULL_ACTION
		else:
			action = np.matmul(weight.T, obs)
			move = np.argmax(action)
		obs, reward, done, info = env.step(move)
		step += 1
		episode_reward += reward
	return episode_reward
	
def main():
	genetic_algorithm()
	create_plot()

if __name__ == "__main__":
	main()
