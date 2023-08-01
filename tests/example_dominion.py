"""
Dominion example/test
"""

from dominion import *
import time

def main():
    # Create a random row of length 12 whose entries are 0, 1, or 2.
    r = random.choices(range(3),k=12)
    # Display this row.
    print(r)
    # Create and display a new row which can follow `r` in a 2-dimensional dominion.
    print(new_row(r,set(range(3))),end='\n\n')

    # Create a random dominion of size 12 whose entries are 0, 1, or 2.
    D = random_dominion(12,set(range(3)))
    # Display that dominion by printing its rows.
    for row in D:
        print(row)
    # The same dominion can be drawn as an image fine as well.
    draw_dominion(D, 'viridis', 'dominion_test' + str(time.time_ns()))

    # print(cm.cmaps_listed)
    # for _ in range(20):
    #     D = random_dominion(20, set(range(3)), random_tree(range(3)))
    #     draw_dominion(D, 'viridis', 'dominion_test' + str(time.time_ns()))


if __name__ == '__main__':
    main()