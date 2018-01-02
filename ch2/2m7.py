import numpy as np

# inspired by https://github.com/cavaunpeu/statistical-rethinking/blob/master/chapter-2/homework.R
#doesn't work for me
# [(b/b, b/w),(b/b,w/w),(b/w,b/b),(b/w,w/w), (w/w,b/b),(w/w,b/w)]
# [2*1, 2*2,0.,1*2,0,0]
ways_producing_state = np.array([2., 4., 0., 2., 0., 0.])
priors = np.repeat(1.,len(ways_producing_state))
posterior = ways_producing_state * priors
posterior = posterior / posterior.sum()

print(posterior)
# the only relevant onse to produce black on other side when first was black
# and other was white are (b/b, b/w),(b/b,w/w)
print('prob that other side is black = ' + str(posterior[0]+posterior[1]))



# inspired by https://rpubs.com/andersgs/my_solutions_chapter2_statrethink
# b1/b2, b1/w1, w1/w2
#
# card |pick|other side
#
# b1/b2+w1/b1| 1 | 1
# b1/b2+b1/w1| 0 | 0
# b2/b1+w1/b1| 1 | 1
# b2/b1+b1/w1| 0 | 0
# b1/b2+w1/w2| 1 | 1
# b1/b2+w2/w1| 1 | 1
# b2/b1+w1/w2| 1 | 1
# b2/b1+w2/w1| 1 | 1
# b1/w1+b1/b2| 0 | 0
# b1/w1+b2/b1| 0 | 0
# b1/w1+w1/w2| 1 | 0
# b1/w1+w2/w1| 1 | 0
# w1/b1+b1/b2| 0 | 0
# w1/b1+b2/b1| 0 | 0
# w1/b1+w1/w2| 0 | 0
# w1/b1+w2/w1| 0 | 0
# w1/w2+b1/b2| 0 | 0
# w1/w2+b2/b1| 0 | 0
# w1/w2+b1/w1| 0 | 0
# w1/w2+w1/b1| 0 | 0
# w2/w1+b1/b2| 0 | 0
# w2/w1+b2/b1| 0 | 0
# w2/w1+b1/w1| 0 | 0
# w2/w1+w1/b1| 0 | 0
#
# to produce black on other side of the first card 6
# from to produce black on the first side of first card and white on second 8
# 6/8 ~ 3/4 ~ 0.75


# bayes theorem
# probability that the second side is black given that the first card side is black
# and second card is white
# 12B - first card has second side black
# 11B - first card has first side black
# 21W - second card has first side white
# 12W - first card has second side white
# P(12B| 11B,21W) = P(11B|12B)*P(21W|12B)*P(12B) / P(11B,21W)
# P(11B,21W) = P(11B|12B)*P(21W|12B)*P(12B) + P(11B|12W)*P(21W|12W)*P(12W)
# hmm
# all posibilities = 24
# P(12B) - 12/24 ~ 1/2 = 0.5
p_12b = 0.5
# P(12W) -12/24 ~ 1/2 = 0.5
p_12w = 0.5
# P(11B|12B) - first card is black given that the second was black
# 12 have second side black, from those 1 side was black 8
# P(11B|12B) = 8/12 ~ 2/3 ~ 0.66
p_11b_12b = 2./3.
# P(21W|12B) - second card has first side white given the second side of first card is black
# 12 have second side black, from those second card was white on first side 8
# P(21W|12B) = 8/12 ~ 0.66
p_21w_12b = 2./3.
# P(11B|12W) - given first card has second white how many of first cards as black
# 12 have second side white, 4 are black on first side
# P(11B|12W) = 4/12 = 1/3
p_11b_12w = 1./3.
# P(21W|12W)  - given the first card has second side white, the second card has first side white
# 12 have second side white, from those second card was white are 4
# P(21W|12W) = 4/12 = 1/3
p_21w_12w = 1./3.
# P(11B,21W) = P(11B|12B)*P(21W|12B)*P(12B) + P(11B|12W)*P(21W|12W)*P(12W)
p_11b_21w = p_11b_12b*p_21w_12b*p_12b + p_11b_12w*p_21w_12w*p_12w
# P(12B| 11B,21W) = P(11B|12B)*P(21W|12B)*P(12B) / P(11B,21W)
p_12b_11b_21w = p_11b_12b*p_21w_12b*p_12b/p_11b_21w

print('result(wrong)=%f'%p_12b_11b_21w)



