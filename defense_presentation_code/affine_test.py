from __future__ import division
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

def draw_object(ax, transformation):
    dish = mpatches.Ellipse([0, 0], 3, 5, ec='k', fc='none', transform=transformation)
    #rod = mlines.Line2D([0, 10], [0, 0], color='k', transform=transformation)
    antenna = mpatches.Circle([10, 0], 1, ec='k', fc='none', transform=transformation)
    ax.add_patch(dish)
    #ax.add_patch(rod)
    ax.add_patch(antenna)

if __name__ == '__main__':
    fig, ax = plt.subplots()
    NoTransform = mtransforms.Affine2D(np.identity(3))
    draw_object(ax, NoTransform)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')
    plt.show()
