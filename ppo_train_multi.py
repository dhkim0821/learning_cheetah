import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

from pybullet_envs.minitaur.envs.minitaur_extended_env import MinitaurExtendedEnv


def create_env(name, render=False):
    if name == 'cartpole':
        env = gym.make('CartPole-v1')
    elif name == 'minitaur':
        env = MinitaurExtendedEnv(history_length=5,
                                  render=render)
    return env


def callback(locals, globals):
    # print('>>>>>>>')
    # for k, v in locals.items():
    #     print('\t', k, v)
    # print('>>>>>>>')
    update = locals['update']
    # print('update: %d' % update)
    if update % 10 == 0:
        print('>>>> callback: save ppo_model')
        model = locals['self']
        model.save("ppo2_model")
        print('>>>> callback: save ppo_model OK')


def learning_rate_func(x):
    print('>>>> learning_rate_func: x = %.6f' % x)
    return 1e-4 * x


if __name__ == '__main__':
    # multiprocess environment
    name = 'minitaur'
    n_cpu = 8
    # env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])
    env = SubprocVecEnv([lambda: create_env(name) for i in range(n_cpu)])

    model = PPO2(MlpPolicy, env,
                 n_steps=1024,
                 learning_rate=learning_rate_func,
                 nminibatches=128,
                 noptepochs=10,
                 verbose=1,
                 tensorboard_log="./tensorboard/%s/" % name)
    model.learn(total_timesteps=1000 * 1000 * 10,
                callback=callback)

    model.save("ppo2_%s" % name)
