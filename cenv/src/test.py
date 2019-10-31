import numpy as np
import cenv

print('add', cenv.add(3, 4))
print('sub', cenv.sub(4, 1))
M = np.random.rand(3, 3)
M = M.transpose().dot(M)
print('inv', cenv.inv(M), np.linalg.inv(M))

env = cenv.PointMass(np.random.rand(2) * 10.0)
print('env.pos = ', env.pos)
print('env.goal = ', env.goal)

for i in range(100):
    k = 1.0
    f = k * (env.goal - env.pos) - np.sqrt(k) * env.vel
    env.apply_force(f)
    print(i, f, env.pos, env.goal, np.linalg.norm(env.pos - env.goal))
