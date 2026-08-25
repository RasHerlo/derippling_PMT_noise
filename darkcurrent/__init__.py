"""Dark-current (no-sample) PMT fringe controls.

Separate from ``batch_defringe``: these recordings are calibration material, not
data to clean. They characterise the fringe layer itself so seeding can verify a
known family instead of searching blindly.

Nothing here writes into experiment folders or the defringe prior cache.
"""

__version__ = "0.1.0"
