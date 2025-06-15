import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp


def init_plot():
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Position Graphs')
    ax.set_title("Y-Position History")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Y (pixels)")
    ax.set_ylim(640, 0)
    ax.grid(True)
    return fig, ax

def update_plot(ax, vals):

    ax.clear()
    ax.set_title("Y-Position History")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Y (pixels)")
    ax.set_ylim(640, 0)
    ax.grid(True)

    for label, y_list in vals.items():
        ax.plot(y_list[:], label=label)

    ax.legend()
    plt.pause(0.001)  # let it refresh










    