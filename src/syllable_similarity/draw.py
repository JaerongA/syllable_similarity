"""
Helper functions for drawing a save_fig
"""

import matplotlib.pyplot as plt

def remove_right_top(ax):
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
