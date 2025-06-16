import numpy as np

class Kalman1D:
    def __init__(self, process_variance=0.01, measurement_variance=0.1):
        self.x = np.array([[0.], [0.]])                 # state = [position, velocity]

        self.P = np.eye(2)                              # Covariance
        self.F = np.array([[1, 1], [0, 1]])             # State Transition Model
        self.H = np.array([[1, 0]])                     # Measurement Model
        self.R = np.array([[measurement_variance]])     # Measurement Noise
        self.Q = process_variance * np.eye(2)           # Process Noise

        self.initialised = False

    def predict(self):
        self.x = self.F @ self.x                        # Prior State Estimate
        self.P = self.F @ self.P @ self.F.T + self.Q    # Prior Covariance Estimate

        return self.x[0, 0]

    def update(self, measurement):
        z = np.array([[measurement]])                   # Sample
        y = z - self.H @ self.x                         # Residual        
        S = self.H @ self.P @ self.H.T + self.R         # Residual Uncertainty
        K = self.P @ self.H.T @ np.linalg.inv(S)        # Kalman Gain

        self.x = self.x + K @ y                         # Posterior State Estimate
        self.P = (np.eye(2) - K @ self.H) @ self.P      # Posterior Covariance Estimate

        self.initialised = True