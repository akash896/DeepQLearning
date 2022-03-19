# This is a sample Python script.
import numpy
import numpy as np
import pygame
import random
import time
import gym
from IPython.display import clear_output
import pygame
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    env = gym.make("FrozenLake8x8-v1")
    print("Done")

    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    q_table = np.zeros((state_space_size, action_space_size))
    print(q_table)

    num_episodes = 100000
    max_steps_per_episodes = 1000

    learning_rate = 0.1
    discount_rate = 0.99

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.0001
    exploration_decay_rate = 0.00001

    rewards_all_episodes = []

    # QLearning algorithm starts
    for episode in range(num_episodes):
        state = env.reset() # reset env state back to starting state at beginning of each episode

        done = False
        rewards_current_episode = 0

        for step in range(max_steps_per_episodes):

            # Exploration and Exploitation trade-off
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state,:])
            else:
                action = env.action_space.sample()

            # getting new state
            new_state, reward, done, info = env.step(action)

            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            state = new_state
            rewards_current_episode += reward

            if done == True:
                break

        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
        rewards_all_episodes.append(rewards_current_episode)

        arr_len = len(rewards_all_episodes)
        parts = arr_len // 10000
        #print("parts = ", parts)
        count = 0
        for i in range(parts):
            add = 0
            for j in range(10000):
                add += rewards_all_episodes[count]
                count += 1
            print(count, " : ", add / 10000)


        # reward_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
        # count = 1000
        # for r in reward_per_thousand_episodes:
        #     print(count, " : ", str(sum(r / 1000)))
        #     count += 1000

        # print("q_table = \n")
        # print(q_table)

    for episode in range(5):
        state = env.reset()
        done = False

        print("***  EPISODE ", episode+1, " *****  \n \n")
        time.sleep(1)
        for step in range(max_steps_per_episodes):
            clear_output(wait=True)
            env.render()
            time.sleep(0.2)

            action = np.argmax(q_table[state,:])
            new_state, reward, done, info = env.step(action)

            if done:
                clear_output(wait=True)
                env.render()
                if reward == 1:
                    print("YOU REACHED THE GOAL")
                    time.sleep(3)
                else:
                    print("yOu FELL INTO A HOLE")
                    time.sleep(1)
                clear_output(wait=True)
                break

            state = new_state
    env.close()



