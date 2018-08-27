from numpy.random import uniform
from math import sqrt

def calculate_pi(n_samples=400):
    counter = 0
    ratio = n_samples/4

    # generate sample points in 2x2 square around origin
    for i in range(n_samples):
        xcoord, ycoord = uniform(-1, 1, 2)

        # increment counter if point falls within radius=1 circle around origin
        if sqrt(xcoord**2 + ycoord**2) < 1:
            counter += 1

    return counter/ratio

if __name__ == '__main__':
    print(calculate_pi(2000000))