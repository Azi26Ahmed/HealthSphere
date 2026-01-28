from .base import BaseVisualizer
from .utils import UtilsVisualizer
from .analysis import AnalysisVisualizer
from .plots_2d import Plots2DVisualizer
from .plots_3d import Plots3DVisualizer
from .plots_ml import MLVisualizer

class Visualizer:
    """
    Main Visualizer class that combines all visualization modules.
    This maintains backward compatibility with the original monolithic class.
    """
    
    def __init__(self):
        """Initialize all visualization modules."""
        # Initialize all sub-visualizers
        self.base = BaseVisualizer()
        self.utils = UtilsVisualizer()
        self.analysis = AnalysisVisualizer()
        self.plots_2d = Plots2DVisualizer()
        self.plots_3d = Plots3DVisualizer()
        self.ml = MLVisualizer()
        
        # Expose common properties from base
        self.disease_options = self.base.disease_options
        self.color_themes = self.base.color_themes
    
    def show_visualization_interface(self):
        """Display the main visualization interface with tabs."""
        return self.base.show_visualization_interface()
    
    # Analysis methods
    def create_overall_health_plot(self, username, disease_type=None):
        return self.analysis.create_overall_health_plot(username, disease_type)
    
    def create_risk_distribution_plot(self, username, disease_type=None):
        return self.analysis.create_risk_distribution_plot(username, disease_type)
    
    def create_prediction_history(self, username, disease_type):
        return self.analysis.create_prediction_history(username, disease_type)
    
    def create_parameter_trends(self, username, disease_type):
        return self.analysis.create_parameter_trends(username, disease_type)
    
    def create_trend_analysis(self, username, disease_type):
        return self.analysis.create_trend_analysis(username, disease_type)
    
    def create_feature_comparison(self, username, disease_type):
        return self.analysis.create_feature_comparison(username, disease_type)
    
    def create_parameter_correlation_plot(self, username, disease_type):
        return self.analysis.create_parameter_correlation_plot(username, disease_type)
    
    def show_synthetic_correlation(self, disease_type):
        return self.analysis.show_synthetic_correlation(disease_type)
    
    def create_disease_risk_timeline(self, username, disease_type):
        return self.analysis.create_disease_risk_timeline(username, disease_type)
    
    # 2D Plot methods
    def create_2d_line_plot(self, username, disease_type):
        return self.plots_2d.create_2d_line_plot(username, disease_type)
    
    def create_2d_scatter_plot(self, username, disease_type):
        return self.plots_2d.create_2d_scatter_plot(username, disease_type)
    
    def create_2d_parameter_plot(self, username, disease_type):
        return self.plots_2d.create_2d_parameter_plot(username, disease_type)
    
    # 3D Plot methods
    def create_3d_scatter_plot(self, username, disease_type):
        return self.plots_3d.create_3d_scatter_plot(username, disease_type)
    
    def create_3d_line_plot(self, username, disease_type):
        return self.plots_3d.create_3d_line_plot(username, disease_type)
    
    def create_3d_surface_plot(self, username, disease_type):
        return self.plots_3d.create_3d_surface_plot(username, disease_type)
    
    def create_3d_mesh_plot(self, username, disease_type):
        return self.plots_3d.create_3d_mesh_plot(username, disease_type)
    
    def create_3d_parameter_plot(self, username, disease_type):
        return self.plots_3d.create_3d_parameter_plot(username, disease_type)
    
    # ML Plot methods
    def create_parameter_impact_plot(self, username, disease_type):
        return self.ml.create_parameter_impact_plot(username, disease_type)
    
    def create_decision_boundary_plot(self, username, disease_type):
        return self.ml.create_decision_boundary_plot(username, disease_type)
    
    def create_feature_importance_plot(self, username, disease_type):
        return self.ml.create_feature_importance_plot(username, disease_type)
    
    def show_sample_parameter_impact(self, disease_type):
        return self.ml.show_sample_parameter_impact(disease_type)
    
    def show_sample_decision_boundary(self, disease_type):
        return self.ml.show_sample_decision_boundary(disease_type)
    
    # Utility methods
    def generate_synthetic_data(self, n_samples=100):
        return self.utils.generate_synthetic_data(n_samples)
    
    def get_prediction_data(self, username):
        return self.base.get_prediction_data(username)

# For backward compatibility, also export the individual classes
__all__ = [
    'Visualizer',
    'BaseVisualizer', 
    'UtilsVisualizer',
    'AnalysisVisualizer',
    'Plots2DVisualizer', 
    'Plots3DVisualizer',
    'MLVisualizer'
] 