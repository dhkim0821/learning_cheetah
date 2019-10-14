import time
import numpy as np
from pybullet_envs.minitaur.envs.minitaur_extended_env import MinitaurExtendedEnv


if __name__ == '__main__':
    print('Hello, PyBullet Gym')
    env = MinitaurExtendedEnv(history_length=1,
                              render=True)

    print('>>> reset..')
    env.reset()
    print('>>> reset OK')
    for i in range(1000):
        ac = np.zeros(8)
        # ac = np.random.rand(8) - 0.5
        tic = time.time()
        ob = env.step(ac)
        toc = time.time()
        print('>>> step %d: (elapsed = %.6fs)' % (i, (toc - tic)))
        env.render()
        time.sleep(0.001)
