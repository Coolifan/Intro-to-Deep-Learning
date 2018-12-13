import numpy as np
left = 0.0
right = 1.0

def discounted_reward(decay, step):
    r = 0.0
    for s in range(step):
        r += decay**s
    return r

while right - left > 1e-8:
    print("Searching [{}, {}]".format(left, right))
    mid = (left + right) / 2.0
    if discounted_reward(mid, 100) < 110*(mid ** 99):
        right = mid
    else:
        left = mid
mid = (left + right) / 2.0
print("gamma = {}. Rewards: {} and {}".format(mid, discounted_reward(mid, 100), 110*(mid ** 99)))
