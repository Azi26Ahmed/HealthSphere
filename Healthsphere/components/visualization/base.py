import streamlit as st
from components.database import get_user_predictions

class BaseVisualizer:
    """
    Base class for visualization components in the HealthSphere app.
    Contains common functionality and interface elements.
    """
    
    def __init__(self):
        """Initialize the BaseVisualizer with plot categories and options."""
        # Define disease options
        self.disease_options = {
            "heart": "Heart Disease",
            "diabetes": "Diabetes",
            "liver": "Liver Disease",
            "kidney": "Kidney Disease",
            "stroke": "Stroke",
            "parkinsons": "Parkinson's Disease",
            "brain_tumor": "Brain Tumor"
        }
        
        # Set default color themes
        self.color_themes = {
            "primary": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
            "pastel": ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462"],
            "diverging": ["#d53e4f", "#fc8d59", "#fee08b", "#e6f598", "#99d594", "#3288bd"]
        }

    def show_visualization_interface(self):
        """Display the main visualization interface with tabs."""
        # Set dark theme
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #262730;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 20px;
            color: white;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1E1E2E;
            color: #4CAF50;
        }
        </style>""", unsafe_allow_html=True)
        
        # Import visualization modules
        from .analysis import AnalysisVisualizer
        from .plots_2d import Plots2DVisualizer
        from .plots_3d import Plots3DVisualizer
        from .plots_ml import MLVisualizer
        
        # Create visualizer instances
        analysis_viz = AnalysisVisualizer()
        plots_2d_viz = Plots2DVisualizer()
        plots_3d_viz = Plots3DVisualizer()
        ml_viz = MLVisualizer()
        
        # Create tabs for different visualization sections
        tabs = st.tabs(["Info", "Visualize"])
        
        # Info tab
        with tabs[0]:
            self.show_info_tab(analysis_viz, plots_2d_viz, plots_3d_viz, ml_viz)
        
        # Visualize tab
        with tabs[1]:
            self.show_visualize_tab(analysis_viz, plots_2d_viz, plots_3d_viz, ml_viz)

    def show_info_tab(self, analysis_viz, plots_2d_viz, plots_3d_viz, ml_viz):
        """Show information analysis visualizations."""
        st.header("Health Data Analysis")
        
        # Get username from session state
        username = st.session_state.get("username", None)
        if not username:
            st.warning("Please log in to access your visualizations.")
            return
        
        # Disease selection (common for all visualizations)
        disease_type = st.selectbox(
            "Select Disease Type", 
            list(self.disease_options.keys()), 
            format_func=lambda x: self.disease_options[x],
            key="info_disease_selector"
        )
        
        # Create analysis type selection directly (without category)
        plot_options = ["Prediction History", "Parameter Trends", "Parameter Correlation", 
                       "Overall Health Analysis", "Trend Analysis", "Feature Comparison"]
        plot_type = st.selectbox("Analysis Type", plot_options, key="info_plot_type")
        
        # Display selected visualization
        st.subheader(f"{plot_type}")
        
        # Generate different visualizations based on selection
        if plot_type == "Prediction History":
            analysis_viz.create_prediction_history(username, disease_type)
        elif plot_type == "Parameter Trends":
            analysis_viz.create_parameter_trends(username, disease_type)
        elif plot_type == "Parameter Correlation":
            analysis_viz.create_parameter_correlation_plot(username, disease_type)
        elif plot_type == "Overall Health Analysis":
            analysis_viz.create_overall_health_plot(username, disease_type)
        elif plot_type == "Trend Analysis":
            analysis_viz.create_trend_analysis(username, disease_type)
        elif plot_type == "Feature Comparison":
            analysis_viz.create_feature_comparison(username, disease_type)

    def show_visualize_tab(self, analysis_viz, plots_2d_viz, plots_3d_viz, ml_viz):
        """Show visualization plots."""
        st.header("Interactive Visualizations")
        
        # Get username from session state
        username = st.session_state.get("username", None)
        if not username:
            st.warning("Please log in to access your visualizations.")
            return
        
        # Disease selection (common for all visualizations)
        disease_type = st.selectbox(
            "Select Disease Type", 
            list(self.disease_options.keys()), 
            format_func=lambda x: self.disease_options[x],
            key="visualize_disease_selector"
        )
        
        # Visualization selection
        st.subheader("Select Visualization Type")
        
        # Create merged plot selection (combining 2D and 3D)
        all_plot_types = ["2D Parameter Plot", "3D Parameter Plot", "2D Scatter Plot", "3D Scatter Plot", 
                          "Disease Risk Timeline", "Parameter Impact", "Decision Boundary", 
                          "Risk Distribution", "Feature Importance"]
        
        plot_type = st.selectbox("Visualization Type", all_plot_types, key="vis_plot_type")
        
        # Display selected visualization
        st.subheader(f"{plot_type}")
        
        # Generate different visualizations based on selection
        if plot_type == "2D Parameter Plot":
            plots_2d_viz.create_2d_parameter_plot(username, disease_type)
        elif plot_type == "3D Parameter Plot":
            plots_3d_viz.create_3d_parameter_plot(username, disease_type)
        elif plot_type == "2D Scatter Plot":
            plots_2d_viz.create_2d_scatter_plot(username, disease_type)
        elif plot_type == "3D Scatter Plot":
            plots_3d_viz.create_3d_scatter_plot(username, disease_type)
        elif plot_type == "Disease Risk Timeline":
            analysis_viz.create_disease_risk_timeline(username, disease_type)
        elif plot_type == "Parameter Impact":
            ml_viz.create_parameter_impact_plot(username, disease_type)
        elif plot_type == "Decision Boundary":
            ml_viz.create_decision_boundary_plot(username, disease_type)
        elif plot_type == "Risk Distribution":
            analysis_viz.create_risk_distribution_plot(username, disease_type)
        elif plot_type == "Feature Importance":
            ml_viz.create_feature_importance_plot(username, disease_type)

    def get_prediction_data(self, username):
        """Get prediction data for a user."""
        return get_user_predictions(username) 