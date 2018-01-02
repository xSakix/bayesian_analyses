# based on https://rpubs.com/andersgs/my_solutions_chapter2_statrethink
# P(twins|A) = 10%
p_twins_a = 0.1
# P(single infant|A) = 90%
p_single_a = 0.9
# P(twins|B) = 20%
p_twins_b = 0.2
# P(single infant| B) = 80%
p_single_b = 0.8
# P(A) = P(B) = 0.5\
p_a, p_b = 0.5, 0.5

# P(twins) = P(twins|A)*P(A) + P(twins|B)*P(B)
p_twins = p_twins_a * p_a + p_twins_b * p_b

# P(A|twins) = P(twins|A)*P(A)/P(twins)
p_a_twins = p_twins_a * p_a / p_twins

# P(B|twins) = P(twins|B)*P(B)/P(twins)
p_b_twins = p_twins_b * p_b / p_twins

#
result = p_a_twins * p_twins_a + p_b_twins * p_twins_b

print('result 2h1 = %f' % (result))

# 2h2 - P(A|twins)
print('result 2h2 = %f' % (p_a_twins))

# 2h3
# P(A | twins, single) = P(twins|A)*P(single|A)*P(A)/P(twins,single)
# P(twins,single) = P(twins|A)*P(single|A)*P(A) + P(twins|B)*P(single|B)*P(B)
p_twins_single = p_twins_a * p_single_a * p_a + p_twins_b * p_single_b * p_b
p_a_twins_single = p_twins_a * p_single_a * p_a / p_twins_single
print('result 2h3=%f' % p_a_twins_single)

# 2h4
p_test_a_a = 0.8
p_test_b_a = 0.2
p_test_b_b = 0.65
p_test_a_b = 0.35

# P(A|testA) = P(testA|A)*P(A)/P(testA)
# P(testA) = P(testA|A)*P(A)+P(testA|B)*P(B)
p_test_a = p_test_a_a * p_a + p_test_a_b * p_b
p_a_test_a = p_test_a_a * p_a / p_test_a
print("result 2h4_1=%f" % p_a_test_a)

# P(A|testA,twins,single) = P(testA|A)*P(twins|A)*P(single|A)*P(A)/P(testA,twins,single)
# P(testA,twins,single) =
#   P(testA|A)*P(twins|A)*P(single|A)*P(A) +
#   P(testA|B)*P(twins|B)*P(single|B)*P(B)

p_test_a_twins_single = p_test_a_a * p_twins_a * p_single_a * p_a + p_test_a_b * p_twins_b * p_single_b * p_b
p_a_testa_twins_single = p_test_a_a * p_twins_a * p_single_a * p_a/p_test_a_twins_single
print('result 2h4_2=%f'%p_a_testa_twins_single)

print('---different approach---')

# based on https://github.com/cavaunpeu/statistical-rethinking/blob/master/chapter-2/homework.R
import numpy as np

like = np.array([p_twins_a, p_twins_b])
prior = np.repeat(1., 2)
post = like * prior
post = post / post.sum()
post_res = post.dot(like)
print("result 2h1=%f" % post_res)
print("result 2h2=%f" % post[0])

like = np.array([p_single_a, p_single_b])
# last posterior is new prior
post = like * post
post = post / post.sum()
print('result 2h3=%f' % post[0])

#2h4-1
like = np.array([p_test_a_a, p_test_a_b])
prior = np.repeat(1., 2)
post = like * prior
post = post / post.sum()
print("result 2h4_1=%f" % post[0])

#2h4-2
# first resolve post from birthing twins
like = np.array([p_twins_a, p_twins_b])
prior = np.repeat(1., 2)
post = like * prior
post = post / post.sum()
like = np.array([p_single_a, p_single_b])
# then resolve posterior after birthing also single
post = like * post
post = post / post.sum()
# then after these events, also resolve the event on testing for A
like = np.array([p_test_a_a, p_test_a_b])
post = like * post
post = post / post.sum()
print("result 2h4_2=%f" % post[0])

