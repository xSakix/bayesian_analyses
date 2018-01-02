import numpy as np

# B/B,B/W,W/W
#
# ways black = b/b, b/b,b/w = 3 ways to produce black from bag~ 3/6 ~ 0.5
# ways_white = b/w,w/w,w/w = 3 ways to produce white~ 0.5
#

# ways when one side is black, that the other side is black -> b/b,b/b
# it cant be b/w from the ways of producing black card, because it would
# produce a white card when turned

#so form all ways a card could be black, those which could produce a black on the\
# other side are 2/3

#[b/b, b/w, w/w]
# inspired by https://github.com/cavaunpeu/statistical-rethinking/blob/master/chapter-2/homework.R
# ways cards can produce first black
ways_producing_black = np.array([2.,1.,0.])

priors = ways_producing_black / ways_producing_black.sum()
print(priors)

#[b/b,b/w,w/w]
# ways cards produce second black
ways_can_produce_black_on_other_side = np.array([2., 0., 0.])
posterior = ways_can_produce_black_on_other_side*priors/ways_can_produce_black_on_other_side.sum()
print(posterior)
print('prob that other side is black = '+str(posterior[0]))
print(posterior[0] == 2./3.)

# inspired by https://rpubs.com/andersgs/my_solutions_chapter2_statrethink
# b1/b2, b1/w1, w1/w2
#
# card |pick|other side
#
# b1/b2| 1 | 1
# b2/b1| 1 | 1
# b1/w1| 1 | 0
# w1/b1| 0 | 0
# w1/w2| 0 | 0
# w2/w1| 0 | 0
#
# to produce black on other side 2
# from to produce black on the first side 3
# 2/3

#bayes theorem
# probability that the second side is black given that the first side is black
# P(2black|1black) = P(1black|2black)*P(2black)/P(1black)
# probability that a card has black on both sides is 1 card from 3 cards
# P(2black) = 1/3
# probability that a card has 1 black side from 6 producing ways of card settings
# is 3, so 3/6 = 1/2
# P(1black) = 1/2
# probability that 1st side is black given the second side is black, e.g.
# that both sides are black is 1
# P(1black|2black) = 1 (hm)
# 1*1/3 / 1/2 ~ 2/3

