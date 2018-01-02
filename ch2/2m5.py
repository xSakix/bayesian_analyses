import numpy as np
# B/B,B/W,W/W,B/B

# inspired by https://github.com/cavaunpeu/statistical-rethinking/blob/master/chapter-2/homework.R
#[b/b, b/w, w/w,b/b]
# in which ways can they produce 1st black (cause there are 2x b/b they can produce 2*2 first black)
ways_producing_black = np.array([4.,1.,0.])

priors = ways_producing_black / ways_producing_black.sum()
print(priors)

#[b/b,b/w,w/w]
# and can produce second black
ways_can_produce_black_on_other_side = np.array([4., 0., 0.])
posterior = ways_can_produce_black_on_other_side*priors/ways_can_produce_black_on_other_side.sum()
print(posterior)
prob_black_other_side = posterior[0]
print('prob that other side is black = ' + str(prob_black_other_side))


# inspired by https://rpubs.com/andersgs/my_solutions_chapter2_statrethink
# b1/b2, b1/w1, w1/w2, b3/b4
#
# card |pick|other side
#
# b1/b2| 1 | 1
# b2/b1| 1 | 1
# b1/w1| 1 | 0
# w1/b1| 0 | 0
# w1/w2| 0 | 0
# w2/w1| 0 | 0
# b3/b4| 1 | 1
# b4/b3| 1 | 1
#
# to produce black on other side 4
# from to produce black on the first side 5
# 4/5 ~ 0.8

#bayes theorem
# probability that the second side is black given that the first side is black
# P(2black|1black) = P(1black|2black)*P(2black)/P(1black)
# probability that a card has black on both sides is 2 card from 4 cards
# P(2black) = 2/4 = 1/2
# probability that a card has 1 black side from 8 producing ways of card settings
# is 5, so 5/8 = 5/8
# P(1black) = 5/8
# probability that 1st side is black given the second side is black, e.g.
# that both sides are black is 1
# P(1black|2black) = 1 (hm)
# 1*1/2 / 5/8 ~ 0.8
