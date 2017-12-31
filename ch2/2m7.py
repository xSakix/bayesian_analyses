import numpy as np

# B/B,B/W,W/W
#
# ways black = b/b, b/b,b/w = 3 ways to produce black from bag~ 3/6 ~ 0.5
#
# 1- b, posibile left b/b,b/w,w/w
#
#

# ways when one side is black, that the other side is black -> b/b,b/b
# it cant be b/w from the ways of producing black card, because it would
# produce a white card when turned

# so form all ways a card could be black, those which could produce a black on the\
# other side are 2/3

# [b/b, b/w, w/w]
ways_producing_state = np.array([2., 2., 2.])

priors = ways_producing_state / ways_producing_state.sum()
print(priors)

# [b/b,b/w,w/w]
ways_can_produce_black_on_other_side = np.array([2., 1., 2.])
posterior = ways_can_produce_black_on_other_side * priors / ways_can_produce_black_on_other_side.sum()
print(posterior)
print('prob that other side is black = ' + str(posterior[0]))

print((2/3*1/3)/0.5)