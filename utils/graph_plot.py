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
    ax.set_ylim(1, 0)
    ax.grid(True)
    return fig, ax

def update_plot(ax, measurements, predictions):
    ax.clear()
    ax.set_title("Y-Position History")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Y (pixels)")
    ax.set_ylim(1, 0)
    ax.grid(True)

    # get measurements and predictions of ball
    y = measurements["Ball"][:,1]
    y_pred = predictions["Ball"][:,1]       # WARNING: y_pred might be shorter than y as cleared after juggle

    # iterate and plot
    for i in range(len(y_pred)):
        if not np.isnan(y[i]):
            ax.plot(i, y[i], color='b', marker="o")
        else:
            ax.plot(i, y_pred[i], color='r', marker="x")

    plt.pause(0.001)  # let it refresh










    