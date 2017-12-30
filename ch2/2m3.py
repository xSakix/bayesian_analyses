import numpy as np

# priors: E 0.7(water) and 0.3(land), M 1.(land)
# observation: L
ways = np.array([1., 0.3])
plausibility = ways/ways.sum()
print('for mars with 1 L observation:%f'%plausibility[0])
print('for earth with 1 L observation:%f'%plausibility[1])


