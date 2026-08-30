# initialize an environment
import gymnasium as gym

# simple env using make() function
# env represents a MDP

env = gym.make('CartPole-v1', render_mode="human")

# reset env to start a new episode
observation, info = env.reset()
print(f"Starting observation: {observation}")

epover = False
totalR = 0

while not epover:
    # choose an action 0 push left, 1 push right
    action = env.action_space.sample()
    # take the action and see what appens
    observation, reward, terminated, truncated, info = env.step(action)
    # reward + 1 for each step the pole stays upright
    # terminate if the pole falls too far
    # truncated if we hit the time limit
    totalR += reward
    epover = terminated or truncated

print(f"Episode finished! Total reward: {totalR}")
env.close()


print(gym.pprint_registry())

'''notes:

Action Space: What can your agent do? (discrete choices, continuous values, etc.) -
Observation Space: What can your agent see? (images, numbers, structured data, etc.)
The goal is to develop a policy - a strategy that tells the agent what action to take in each situation to maximize long-term rewards.
Bellman equation in action - it says the value of a state-action pair should equal the immediate reward plus the discounted value of the best next action.
'''