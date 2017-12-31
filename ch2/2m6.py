import numpy as np

# B/B,B/W,W/W
#

# ways to produce black = 1 b/b,1 b/b, b/w 2 = 4 ways
# from those only 2 can produce black when turned ~ 2/4 = 0.5

# [b/b, b/w, w/w]
ways_producing_black = np.array([2., 2., 0.])

priors = ways_producing_black / ways_producing_black.sum()
print(priors)

# [b/b,b/w,w/w]
ways_can_produce_black_on_other_side = np.array([2., 0., 0.])
posterior = ways_can_produce_black_on_other_side * priors / ways_can_produce_black_on_other_side.sum()
print(posterior)
print('prob that other side is black = ' + str(posterior[0]))
