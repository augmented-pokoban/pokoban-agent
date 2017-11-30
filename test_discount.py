import random
import helper

sample = random.sample(range(-10, 10), 10)

print(sample)

print(helper.discount(sample, 0.99))
print(helper.discount_old(sample, 0.99))