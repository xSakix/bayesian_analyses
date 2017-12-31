import numpy as np
# B/B,B/W,W/W,B/B
#
# ways black = b/b,b/b,b/w = 3 ways to produce black from bag
# adding another b/b doesnt add a new way of producing black - it's stil the
# same type of card

# ways when one side is black, that the other side is black -> b/b,b/b
# it cant be b/w from the ways of producing black card, because it would
# produce a white card when turned

#so form all ways a card could be black, those which could produce a black on the\
# other side are 2/3


#[b/b, b/w, w/w,b/b]
ways_producing_black = np.array([2.,1.,0.,2.])

priors = ways_producing_black / ways_producing_black.sum()
print(priors)

#[b/b,b/w,w/w]
ways_can_produce_black_on_other_side = np.array([2., 0., 0.,2.])
posterior = ways_can_produce_black_on_other_side*priors/ways_can_produce_black_on_other_side.sum()
print(posterior)
prob_black_other_side = posterior[0] + posterior[3]
print('prob that other side is black = ' + str(prob_black_other_side))
