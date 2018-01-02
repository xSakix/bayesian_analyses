import numpy as np

# B/B,B/W,W/W
#

# ways to produce black = 1 b/b,1 b/b, b/w 2 = 4 ways
# from those only 2 can produce black when turned ~ 2/4 = 0.5

# inspired by https://github.com/cavaunpeu/statistical-rethinking/blob/master/chapter-2/homework.R
# [b/b, b/w, w/w]
#in which ways can the cards produce 1st black
ways_producing_black = np.array([2., 2., 0.])

priors = ways_producing_black / ways_producing_black.sum()
print(priors)

# [b/b,b/w,w/w]
# in which ways can they then produce second black
ways_can_produce_black_on_other_side = np.array([2., 0., 0.])
posterior = ways_can_produce_black_on_other_side * priors / ways_can_produce_black_on_other_side.sum()
print(posterior)
print('prob that other side is black = ' + str(posterior[0]))
# inspired by https://rpubs.com/andersgs/my_solutions_chapter2_statrethink
# b1/b2, b1/w1, w1/w2
#
# card |pick|other side
#
# b1/b2| 1 | 1
# b2/b1| 1 | 1
# b1/w1| 1 | 0
# w1/b1| 0 | 0
# b1/w1| 1 | 0
# w1/b1| 0 | 0
# w1/w2| 0 | 0
# w2/w1| 0 | 0
# w1/w2| 0 | 0
# w2/w1| 0 | 0
# w1/w2| 0 | 0
# w2/w1| 0 | 0
#
# to produce black on other side 4
# from to produce black on the first side 2
# 2/4 ~ 0.5


#bayes theorem
# probability that the second side is black given that the first side is black
# P(2black|1black) = P(1black|2black)*P(2black)/P(1black)
# probability that a card has black on both sides is 2 card from 12 ways
# P(2black) = 2/12 = 1/6
# probability that a card has 1st black side from 12 producing ways of card settings
# is 4, so 4/12 = 1/3
# P(1black) = 1/3
# probability that 1st side is black given the second side is black, e.g.
# that both sides are black is 1
# P(1black|2black) = 1 (hm)
# 1*1/6 / 1/3 ~ 1/2 ~ 0.5
