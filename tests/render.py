import gymnasium as gym
import time

import numpy as np

import gym_hybrid


if __name__ == '__main__':
    env = gym.make('Moving-v0', render_mode='human')
    env.reset()

    done = False
    rtgs = []
    rewards = []
    backwards_returns = []
    backwards_return = 0
    gamma = 0.99
    step = 0

    running_mean = 0
    running_var = 0
    epsilon = 1e-8
    norm_rewards = []

    while not done:
        _, reward, done, _, _ = env.step(env.action_space.sample())

        reward = 5 * reward
        rewards.append(reward)

        running_mean = np.array(rewards).mean()
        running_var = np.array(rewards).var()
        norm_rewards.append((reward - running_mean) / np.sqrt(running_var + epsilon))
        print("Normalized Reward: ", (reward - running_mean) / np.sqrt(running_var + epsilon))

        backwards_return = reward + gamma * backwards_return
        backwards_returns.append(backwards_return)
        for i in range(len(rtgs)):
            rtgs[-i-1] += gamma ** (i + 1) * reward

        rtgs.append(reward)
        step += 1
        env.render()
        # time.sleep(0.1)
        # print("Reward: ", reward)

    print("RTGs: ", rtgs)
    print("Rewards: ", rewards)
    print("Normalized Rewards mean: ", np.array(norm_rewards).mean())
    print("Normalized Rewards var: ", np.array(norm_rewards).var())
    print("Backwards returns: ", backwards_returns)
    print("RTGs which is greater than 0: ", [rtg for rtg in rtgs if rtg > 0])
    print("Step: ", step)

    time.sleep(1)
    env.close()

    # plot the rewards, rtgs, and backwards returns
    import matplotlib.pyplot as plt
    plt.plot(rewards, label='Rewards', color='red', marker='o')
    plt.plot(rtgs, label='RTGs', color='blue')
    plt.plot(backwards_returns, label='Backwards Returns', color='green')

    plt.xlabel('Step')
    plt.legend()

    plt.show()

    # normalize the rtgs
    var1 = np.array(rtgs).var()
    print("Variance of RTGs: ", var1)
    plt.plot(rtgs, label='RTGs', color='blue')
    plt.plot(rtgs / np.sqrt(var1), label='Normalized by RTGs', color='blue', linestyle='dashed')
    print("Normalized RTGs: ", np.array(rtgs / np.sqrt(var1)).var())
    var2 = np.array(backwards_returns).var()
    print("Variance of Backwards Returns: ", var2)
    print("Normalized Backwards Returns: ", np.array(rtgs / np.sqrt(var2)).var())
    plt.plot(rtgs / np.sqrt(var2), label='Normalized by Backwards Returns', color='green', linestyle='dashed')

    plt.xlabel('Step')
    plt.legend()

    plt.show()
