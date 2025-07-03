import matplotlib.pyplot as plt
import numpy as np


def init_landmark_plot(title="Landmark Position History", figsize=(12, 8)):
    """
    Initialize a plot for landmark tracking with flexible configuration
    
    Args:
        title: Title for the plot window
        figsize: Figure size (width, height)
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.manager.set_window_title(title)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Position (normalized)")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()
    return fig, ax


def plot_landmark_timeseries(measurements, predictions, landmark_name, 
                           coordinate='both', show_predictions=True, 
                           ax=None, figsize=(12, 6)):
    """
    Plot time series of a specific landmark
    
    Args:
        measurements: Dictionary of measurement data
        predictions: Dictionary of prediction data
        landmark_name: Name of the landmark to plot (e.g., 'Head', 'Left_Knee')
        coordinate: Which coordinate to plot ('x', 'y', or 'both')
        show_predictions: Whether to show prediction data
        ax: Matplotlib axis (if None, creates new plot)
        figsize: Figure size for new plots
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{landmark_name} Position Over Time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Position (normalized)")
        ax.grid(True)
    else:
        fig = ax.get_figure()
    
    if landmark_name not in measurements:
        print(f"Warning: {landmark_name} not found in measurements")
        return fig, ax
    
    # Get data
    data = measurements[landmark_name]
    if len(data) == 0:
        print(f"Warning: No data for {landmark_name}")
        return fig, ax
    
    frames = np.arange(len(data))
    
    # Plot based on coordinate selection
    if coordinate in ['x', 'both']:
        x_vals = data[:, 0]
        valid_x = ~np.isnan(x_vals)
        ax.plot(frames[valid_x], x_vals[valid_x], 'b-', marker='o', 
                markersize=3, label=f'{landmark_name} X', alpha=0.7)
        
        if show_predictions and landmark_name in predictions:
            pred_data = predictions[landmark_name]
            if len(pred_data) > 0:
                pred_frames = np.arange(len(pred_data))
                pred_x = pred_data[:, 0]
                valid_pred_x = ~np.isnan(pred_x)
                ax.plot(pred_frames[valid_pred_x], pred_x[valid_pred_x], 
                       'r--', marker='x', markersize=3, 
                       label=f'{landmark_name} X (predicted)', alpha=0.7)
    
    if coordinate in ['y', 'both']:
        y_vals = data[:, 1]
        valid_y = ~np.isnan(y_vals)
        color = 'g' if coordinate == 'both' else 'b'
        ax.plot(frames[valid_y], y_vals[valid_y], f'{color}-', marker='o', 
                markersize=3, label=f'{landmark_name} Y', alpha=0.7)
        
        if show_predictions and landmark_name in predictions:
            pred_data = predictions[landmark_name]
            if len(pred_data) > 0:
                pred_frames = np.arange(len(pred_data))
                pred_y = pred_data[:, 1]
                valid_pred_y = ~np.isnan(pred_y)
                pred_color = 'orange' if coordinate == 'both' else 'r'
                ax.plot(pred_frames[valid_pred_y], pred_y[valid_pred_y], 
                       f'{pred_color}--', marker='x', markersize=3, 
                       label=f'{landmark_name} Y (predicted)', alpha=0.7)
    
    ax.legend()
    ax.set_ylim(0, 1)
    
    return fig, ax


def plot_multiple_landmarks(measurements, predictions, landmark_names, 
                          coordinate='x', show_predictions=False, 
                          figsize=(15, 8)):
    """
    Plot multiple landmarks on the same graph for comparison
    
    Args:
        measurements: Dictionary of measurement data
        predictions: Dictionary of prediction data
        landmark_names: List of landmark names to plot
        coordinate: Which coordinate to plot ('x' or 'y')
        show_predictions: Whether to show prediction data
        figsize: Figure size
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f"Multiple Landmarks - {coordinate.upper()} Position Comparison")
    ax.set_xlabel("Frame")
    ax.set_ylabel(f"{coordinate.upper()} Position (normalized)")
    ax.grid(True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(landmark_names)))
    coord_idx = 0 if coordinate == 'x' else 1
    
    for i, landmark_name in enumerate(landmark_names):
        if landmark_name not in measurements:
            print(f"Warning: {landmark_name} not found in measurements")
            continue
            
        data = measurements[landmark_name]
        if len(data) == 0:
            continue
            
        frames = np.arange(len(data))
        vals = data[:, coord_idx]
        valid = ~np.isnan(vals)
        
        ax.plot(frames[valid], vals[valid], color=colors[i], 
                marker='o', markersize=2, label=landmark_name, alpha=0.8)
        
        if show_predictions and landmark_name in predictions:
            pred_data = predictions[landmark_name]
            if len(pred_data) > 0:
                pred_frames = np.arange(len(pred_data))
                pred_vals = pred_data[:, coord_idx]
                valid_pred = ~np.isnan(pred_vals)
                ax.plot(pred_frames[valid_pred], pred_vals[valid_pred], 
                       color=colors[i], linestyle='--', marker='x', 
                       markersize=2, label=f'{landmark_name} (pred)', alpha=0.6)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    
    return fig, ax


def update_landmark_plot(ax, measurements, predictions, landmark_names, 
                        coordinate='x', show_predictions=False):
    """
    Update an existing landmark plot with new data (for real-time plotting)
    
    Args:
        ax: Matplotlib axis to update
        measurements: Dictionary of measurement data
        predictions: Dictionary of prediction data
        landmark_names: List of landmark names to plot
        coordinate: Which coordinate to plot ('x' or 'y')
        show_predictions: Whether to show prediction data
    """
    ax.clear()
    ax.set_title(f"Multiple Landmarks - {coordinate.upper()} Position (Real-time)")
    ax.set_xlabel("Frame")
    ax.set_ylabel(f"{coordinate.upper()} Position (normalized)")
    ax.grid(True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(landmark_names)))
    coord_idx = 0 if coordinate == 'x' else 1
    
    for i, landmark_name in enumerate(landmark_names):
        if landmark_name not in measurements:
            continue
            
        data = measurements[landmark_name]
        if len(data) == 0:
            continue
            
        frames = np.arange(len(data))
        vals = data[:, coord_idx]
        valid = ~np.isnan(vals)
        
        if np.any(valid):
            ax.plot(frames[valid], vals[valid], color=colors[i], 
                    marker='o', markersize=2, label=landmark_name, alpha=0.8)
        
        if show_predictions and landmark_name in predictions:
            pred_data = predictions[landmark_name]
            if len(pred_data) > 0:
                pred_frames = np.arange(len(pred_data))
                pred_vals = pred_data[:, coord_idx]
                valid_pred = ~np.isnan(pred_vals)
                if np.any(valid_pred):
                    ax.plot(pred_frames[valid_pred], pred_vals[valid_pred], 
                           color=colors[i], linestyle='--', marker='x', 
                           markersize=2, label=f'{landmark_name} (pred)', alpha=0.6)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1)
    plt.pause(0.001)


def save_landmark_plot(measurements, predictions, landmark_names, 
                      coordinate='y', filename='landmark_plot.png', 
                      show_predictions=False):
    """
    Save a landmark plot to file
    
    Args:
        measurements: Dictionary of measurement data
        predictions: Dictionary of prediction data
        landmark_names: List of landmark names to plot
        coordinate: Which coordinate to plot ('x' or 'y')
        filename: Output filename
        show_predictions: Whether to show prediction data
    """
    fig, ax = plot_multiple_landmarks(measurements, predictions, landmark_names, 
                                    coordinate, show_predictions)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {filename}")










    