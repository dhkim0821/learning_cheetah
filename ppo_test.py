import gym
from stable_baselines import PPO2
from ppo_train_multi import create_env
import pybullet as p


name = 'pointmass'
env = create_env(name, render=True)
# env = gym.make('CartPole-v1')
# Load the trained agent
model = PPO2.load('ppo2_model')

# p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "test.mp4")

# Enjoy trained agent
obs = env.reset()
sum_rewards = 0.0
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    sum_rewards += rewards
    env.render()
    # sim_time =  env.minitaur.GetTimeSinceReset()
    sim_time = env.env.t
    print('%d (%.4fs): %.4f (%.4f)' % (i, sim_time, rewards, sum_rewards))
    if dones:
        break
