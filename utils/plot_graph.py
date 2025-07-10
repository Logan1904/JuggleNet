# utils/plot_graph.py
import matplotlib.pyplot as plt
import numpy as np

# Global variables to maintain plot history
plot_history = {
    'measurements': [],
    'predictions': [],
    'errors': []
}
MAX_PLOT_BUFFER = 100

def init_plot():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.canvas.manager.set_window_title('Position Graphs')
    
    ax1.set_title("Y-Position History")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Y (pixels)")
    ax1.set_ylim(1, 0)
    ax1.grid(True)
    
    ax2.set_title("Kalman Filter Prediction Error")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Error (pixels)")
    ax2.grid(True)
    
    return fig, (ax1, ax2)

def update_plot(axes, measurements, predictions):
    ax1, ax2 = axes
    
    # get measurements and predictions of ball
    y = measurements["Ball"][:,1]
    y_pred = predictions["Ball"][:,1]

    # add latest values to our internal persistant buffer
    plot_history['measurements'].append(y[-1])
    plot_history['predictions'].append(y_pred[-1])

    if not np.isnan(y[-1]) or not np.isnan([y_pred[-1]]):
        error = abs(y[-1]-y_pred[-1])
    else:
        error = np.nan
    plot_history['errors'].append(error)
     
    # Trim history to max buffer length
    if len(plot_history['measurements']) > MAX_PLOT_BUFFER:
        plot_history['measurements'] = plot_history['measurements'][-MAX_PLOT_BUFFER:]
        plot_history['predictions'] = plot_history['predictions'][-MAX_PLOT_BUFFER:]
        plot_history['errors'] = plot_history['errors'][-MAX_PLOT_BUFFER:]

    # Clear and redraw plots
    ax1.clear()
    ax1.set_title("Y-Position History")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Y (pixels)")
    ax1.set_ylim(1, 0)
    ax1.grid(True)
    
    ax2.clear()
    ax2.set_title("Kalman Filter Prediction Error")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Error (pixels)")
    ax1.set_ylim(1, 0)
    ax2.grid(True)
    
    # Plot position history
    for i, (meas, pred) in enumerate(zip(plot_history['measurements'], plot_history['predictions'])):
        if not np.isnan(meas):
            ax1.plot(i, meas, color='b', marker="o", label='Measurement' if i == 0 else "")
        if not np.isnan(pred):
            ax1.plot(i, pred, color='r', marker="x", label='Prediction' if i == 0 else "")
    
    # Plot error history
    for i, error in enumerate(plot_history['errors']):
        if not np.isnan(error):
            ax2.plot(i, error, color='g', marker="o")
    
    # Add legends
    ax1.legend()
    
    plt.tight_layout()
    plt.pause(0.001)  # let it refresh