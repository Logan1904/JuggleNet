from scipy.signal import find_peaks
import numpy as np


class JuggleCounter:
    """
    Detects juggle events from ball trajectory using peak detection.

    Y-coordinate increases as you move lower down the screen. Therefore, 
    a minimum height in a ball trajectory, corresponding to a juggle kick,
    manifests as a maximum point in the Y-coordinate.

    We maintain state across frames to prevent double-counting and handle
    sliding window history.
    """

    def __init__(self,
                 prominence: float = 0.02,
                 min_gap_frames: int = 5,
                 max_distance: float = 0.3,
                 min_history_length: int = 10):
        """
        Args:
            prominence: Minimum prominence for peak detection
            min_gap_frames: Minimum frames between valid juggles
            max_distance: Maximum normalised distance for valid juggle attribution
            min_history_length: Minimum trajectory points needed for detection
        """
        
        self.prominence = prominence
        self.min_gap_frames = min_gap_frames
        self.max_distance = max_distance
        self.min_history_length = min_history_length

        # State tracking
        self.last_counted_peak_idx = -np.inf
        self.counts = {}
        self.frame_count = 0

    def update(self, predictions: dict) -> dict:
        """
        Update juggle count based on current predictions
        
        Args:
            predictions: Dict of {POI: np.array(N, 4)}, with trajectory history
                         Format: [x, y, width, height] per row

        Returns:
            Updated self.counts dictionary
        """
        
        if "Ball" not in predictions or len(predictions["Ball"]) < self.min_history_length:
            return self.counts
        
        # Extract ball y-coordinates
        y = predictions["Ball"][:, 1]

        # Find all peaks in trajectory
        peaks, properties = find_peaks(y, prominence=self.prominence)

        # Process each peak
        for peak_idx in peaks:
            # Check if this is a new peak (not already counted, sufficient time gap)
            if self.is_valid_peak(peak_idx):
                # Find closest body part and count juggle
                self.count_juggle_at_peak(peak_idx, predictions)
                self.last_counted_peak_idx = peak_idx


    def is_valid_peak(self, peak_idx: int) -> bool:
        """Check if peak should be counted (not double-counted, sufficient time gap)"""
        return peak_idx > self.last_counted_peak_idx + self.min_gap_frames
    
    def count_juggle_at_peak(self, peak_idx: int, predictions: dict):
        """Find closest body part at peak and increment count"""
        
        ball_pos = predictions["Ball"][peak_idx, :2]    # x, y coordinates

        min_dist = np.inf
        closest_part = None

        # Find closest body part
        for key, value in predictions.items():
            if key == "Ball" or len(value) <= peak_idx:
                continue
            
            part_pos = value[peak_idx, :2]
            dist = np.linalg.norm(ball_pos - part_pos)

            if dist < min_dist:
                min_dist = dist
                closest_part = key
        
        # Only count if within distance threshold
        if closest_part and min_dist < self.max_distance:
            if closest_part not in self.counts:
                self.counts[closest_part] = 0
            self.counts[closest_part] += 1
    
    def reset(self):
        """Reset all counts and state"""
        self.counts = {}
        self.last_counted_peak_idx = np.inf
        self.frame_count = 0

    def get_counts(self) -> dict:
        """Get current juggle count"""
        return self.counts.copy()
    