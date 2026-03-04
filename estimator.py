"""
estimator.py
============
Broyden Rank-1 Jacobian Estimator with Singularity Projection.
Safeguarded against initial contact transients and zero-crossing explosions.
"""

import numpy as np

class JacobianEstimator:
    def __init__(self, alpha=0.6):
        # Initialize with a strong positive guess
        self.J_est = np.eye(2) 
        self.alpha = alpha
        
        self.q_prev = None
        self.x_prev = None

        # Singularity Safeguard: J11 must never drop below this value
        self.J_MIN = 0.1  

    def update(self, q_curr, x_curr, time_s=0.0):
        # 1. DEADBAND: Do not learn during the chaotic drop/initialization phase
        if time_s < 1.0:
            self.q_prev = q_curr.copy()
            self.x_prev = x_curr.copy()
            return self.J_est

        if self.q_prev is None:
            self.q_prev = q_curr.copy()
            self.x_prev = x_curr.copy()
            return self.J_est

        delta_q = q_curr - self.q_prev
        delta_x = x_curr - self.x_prev

        # 2. BROYDEN UPDATE
        if np.linalg.norm(delta_q) > 1e-5:
            pred_dx = self.J_est @ delta_q
            error = delta_x - pred_dx

            numerator = np.outer(error, delta_q)
            # 1e-6 prevents division by zero if delta_q is extremely small
            denominator = np.dot(delta_q, delta_q) + 1e-6
            
            self.J_est = self.J_est + self.alpha * (numerator / denominator)

        # 3. SINGULARITY PROJECTION (The Red Pill Fix)
        # Force the AI to obey physical reality: Pulling left MUST move left.
        # If the math tries to invert the Jacobian, we hard-clamp it to J_MIN
        # to prevent the Control Law from dividing by zero and detonating the FEM solver.
        if self.J_est[0, 0] < self.J_MIN:
            self.J_est[0, 0] = self.J_MIN

        self.q_prev = q_curr.copy()
        self.x_prev = x_curr.copy()
        
        return self.J_est