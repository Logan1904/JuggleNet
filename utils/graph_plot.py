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

def update_plot(ax, history_dict):

    ax.clear()
    ax.set_title("Y-Position History")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Y (pixels)")
    ax.set_ylim(640, 0)
    ax.grid(True)


    for label, y_list in history_dict.items():
        ax.plot(y_list[:], label=label)
        break

    ax.legend()
    plt.pause(0.001)  # let it refresh

    return history_dict










    