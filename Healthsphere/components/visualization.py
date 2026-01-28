# This file maintains backward compatibility by importing the modular Visualizer
# The original large visualization.py has been split into smaller modules:
# - base.py: Core functionality and interface
# - utils.py: Utility functions and data generation
# - analysis.py: Statistical analysis and health trend visualizations
# - plots_2d.py: 2D plotting methods
# - plots_3d.py: 3D plotting methods  
# - plots_ml.py: Machine learning visualization methods

from .visualization import Visualizer

# For backward compatibility, expose the Visualizer class at the module level
__all__ = ['Visualizer'] 