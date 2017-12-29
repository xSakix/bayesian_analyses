import grid_approx
import matplotlib.pyplot as plt

#for W
# 1
p_grid, posterior = grid_approx.compute(['W', 'W', 'W'])
plt.plot(p_grid, posterior, label='#1')

# 2
p_grid, posterior = grid_approx.compute(['W', 'W', 'W', 'L'])
plt.plot(p_grid, posterior, label='#2')
# 3
p_grid, posterior = grid_approx.compute(['L', 'W', 'W', 'L', 'W', 'W', 'W'])
plt.plot(p_grid, posterior, label='#3')

plt.xlabel('probability of W')
plt.ylabel('plausibility')
plt.legend()
plt.show()
plt.clf()

#for L
# 1
p_grid, posterior = grid_approx.compute(['W', 'W', 'W'],'L')
plt.plot(p_grid, posterior, label='#1')
# 2
p_grid, posterior = grid_approx.compute(['W', 'W', 'W', 'L'],'L')
plt.plot(p_grid, posterior, label='#2')
# 3
p_grid, posterior = grid_approx.compute(['L', 'W', 'W', 'L', 'W', 'W', 'W'],'L')
plt.plot(p_grid, posterior, label='#3')

plt.xlabel('probability of L')
plt.ylabel('plausibility')
plt.legend()
plt.show()

# 4
for j in range(4):
    obs = ['L', 'W', 'W', 'L', 'W', 'W', 'W']
    for i in range(j*10):
        obs.append(obs)

    p_grid,posterior = grid_approx.compute(obs)
    plt.plot(p_grid, posterior, label='#' + str(len(obs)))


plt.xlabel('probability of W')
plt.ylabel('plausibility')
plt.legend()
plt.show()

# 4L
for j in range(4):
    obs = ['L', 'W', 'W', 'L', 'W', 'W', 'W']
    for i in range(j * 10):
        obs.append(obs)

    p_grid, posterior = grid_approx.compute(obs,'L')
    plt.plot(p_grid, posterior, label='#' + str(len(obs)))

plt.xlabel('probability of L')
plt.ylabel('plausibility')
plt.legend()
plt.show()

