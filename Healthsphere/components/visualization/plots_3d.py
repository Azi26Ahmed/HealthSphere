import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from components.database import get_user_prediction_parameters

class Plots3DVisualizer:
    """
    3D visualization class for health data plotting.
    Contains methods for 3D scatter plots, line plots, surface plots, and mesh plots.
    """
    
    def __init__(self):
        self.disease_options = {
            "heart": "Heart Disease",
            "diabetes": "Diabetes",
            "liver": "Liver Disease",
            "kidney": "Kidney Disease",
            "stroke": "Stroke",
            "parkinsons": "Parkinson's Disease",
            "brain_tumor": "Brain Tumor"
        }

    def create_3d_parameter_plot(self, username, disease_type):
        """Main 3D parameter plot method - wrapper for 3D scatter plot."""
        self.create_3d_scatter_plot(username, disease_type)

    def create_3d_scatter_plot(self, username, disease_type):
        """Create a customizable 3D scatter plot."""
        # Get parameter data
        param_data = get_user_prediction_parameters(username, disease_type)
        
        if not param_data or len(param_data) == 0:
            st.info(f"No parameter data available for {self.disease_options[disease_type]}.")
            return
        
        # Extract parameters
        parameters = {}
        
        for record in param_data:
            if 'input_data' in record and 'timestamp' in record:
                timestamp = record['timestamp']
                
                for param, value in record['input_data'].items():
                    try:
                        # Try to convert to float
                        value = float(value)
                        if param not in parameters:
                            parameters[param] = []
                        parameters[param].append((timestamp, value))
                    except (ValueError, TypeError):
                        # Skip non-numeric parameters
                        continue
        
        if not parameters:
            st.info("No numeric parameters found for visualization.")
            return
        
        # Need at least 3 parameters for 3D visualization
        if len(parameters) < 3:
            st.info("Need at least 3 numeric parameters for 3D visualization.")
            return
        
        # Allow user to select parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_param = st.selectbox("X-axis Parameter", list(parameters.keys()), index=0, key="3d_scatter_x")
        
        with col2:
            y_param = st.selectbox("Y-axis Parameter", list(parameters.keys()), 
                                 index=min(1, len(parameters.keys())-1), key="3d_scatter_y")
        
        with col3:
            z_param = st.selectbox("Z-axis Parameter", list(parameters.keys()), 
                                 index=min(2, len(parameters.keys())-1), key="3d_scatter_z")
        
        # Create the visualization
        if x_param in parameters and y_param in parameters and z_param in parameters:
            # Extract parameter data
            x_data = dict(parameters[x_param])
            y_data = dict(parameters[y_param])
            z_data = dict(parameters[z_param])
            
            # Find common timestamps
            common_timestamps = set(x_data.keys()).intersection(
                set(y_data.keys())).intersection(set(z_data.keys()))
            
            if len(common_timestamps) < 2:
                st.info(f"Not enough matching data points for {x_param}, {y_param}, and {z_param}.")
                st.info("Even with minimal data, 3D plots require at least 2 data points.")
                return
                
            # Create data for plotting
            x_values = [x_data[ts] for ts in common_timestamps]
            y_values = [y_data[ts] for ts in common_timestamps]
            z_values = [z_data[ts] for ts in common_timestamps]
            
            # Create 3D scatter plot
            fig = go.Figure(data=[go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode='markers',
                marker=dict(
                    size=10,
                    color=z_values,
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=[str(ts) for ts in common_timestamps]
            )])
            
            fig.update_layout(
                title=f'3D Parameter Visualization for {self.disease_options[disease_type]}',
                scene=dict(
                    xaxis_title=x_param,
                    yaxis_title=y_param,
                    zaxis_title=z_param,
                    xaxis=dict(backgroundcolor="rgba(30, 30, 46, 0.8)"),
                    yaxis=dict(backgroundcolor="rgba(30, 30, 46, 0.8)"),
                    zaxis=dict(backgroundcolor="rgba(30, 30, 46, 0.8)")
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select valid parameters for visualization.")

    def create_3d_line_plot(self, username, disease_type):
        """Create a 3D line plot showing parameter evolution over time."""
        # Get parameter data
        param_data = get_user_prediction_parameters(username, disease_type)
        
        if not param_data or len(param_data) == 0:
            st.info(f"No parameter data available for {self.disease_options[disease_type]}.")
            return
        
        # Extract parameters
        parameters = {}
        timestamps = []
        
        for record in param_data:
            if 'input_data' in record and 'timestamp' in record:
                timestamp = record['timestamp']
                timestamps.append(timestamp)
                
                for param, value in record['input_data'].items():
                    try:
                        # Try to convert to float
                        value = float(value)
                        if param not in parameters:
                            parameters[param] = []
                        parameters[param].append((timestamp, value))
                    except (ValueError, TypeError):
                        # Skip non-numeric parameters
                        continue
        
        if not parameters:
            st.info("No numeric parameters found for visualization.")
            return
        
        # Need at least 3 parameters for 3D visualization
        if len(parameters) < 3:
            st.info("Need at least 3 numeric parameters for 3D visualization.")
            return
        
        # Need timestamps sorted chronologically
        timestamps = sorted(list(set(timestamps)))
        
        # Allow user to select parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_param = st.selectbox("X-axis Parameter", list(parameters.keys()), index=0, key="3d_line_x")
        
        with col2:
            y_param = st.selectbox("Y-axis Parameter", list(parameters.keys()), 
                                index=min(1, len(parameters.keys())-1), key="3d_line_y")
        
        with col3:
            z_param = st.selectbox("Z-axis Parameter", list(parameters.keys()), 
                                index=min(2, len(parameters.keys())-1), key="3d_line_z")
        
        # Create the visualization
        if x_param in parameters and y_param in parameters and z_param in parameters:
            # Create data dictionaries for easier lookup
            x_dict = dict(parameters[x_param])
            y_dict = dict(parameters[y_param])
            z_dict = dict(parameters[z_param])
            
            # Find common timestamps
            common_timestamps = set(x_dict.keys()).intersection(
                set(y_dict.keys())).intersection(set(z_dict.keys()))
            
            if len(common_timestamps) < 2:
                st.info(f"Not enough matching data points for {x_param}, {y_param}, and {z_param}.")
                st.info("Need at least 2 data points for line plots.")
                return
            
            # Sort timestamps
            common_ts_sorted = sorted(common_timestamps)
            
            # Create data for plotting
            x_values = [x_dict[ts] for ts in common_ts_sorted]
            y_values = [y_dict[ts] for ts in common_ts_sorted]
            z_values = [z_dict[ts] for ts in common_ts_sorted]
            
            # Create 3D line plot
            fig = go.Figure(data=[go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode='lines+markers',
                line=dict(
                    color='#1f77b4',
                    width=4
                ),
                marker=dict(
                    size=6,
                    color=list(range(len(common_ts_sorted))),
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=[str(ts) for ts in common_ts_sorted]
            )])
            
            # Add annotations for the start and end points
            fig.add_trace(go.Scatter3d(
                x=[x_values[0]],
                y=[y_values[0]],
                z=[z_values[0]],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color='green',
                    symbol='circle'
                ),
                text=['Start'],
                textposition='top center',
                name='Start Point'
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[x_values[-1]],
                y=[y_values[-1]],
                z=[z_values[-1]],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='circle'
                ),
                text=['End'],
                textposition='top center',
                name='End Point'
            ))
            
            fig.update_layout(
                title=f'3D Parameter Evolution for {self.disease_options[disease_type]}',
                scene=dict(
                    xaxis_title=x_param,
                    yaxis_title=y_param,
                    zaxis_title=z_param,
                    xaxis=dict(backgroundcolor="rgba(30, 30, 46, 0.8)"),
                    yaxis=dict(backgroundcolor="rgba(30, 30, 46, 0.8)"),
                    zaxis=dict(backgroundcolor="rgba(30, 30, 46, 0.8)")
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select valid parameters for visualization.")

    def create_3d_surface_plot(self, username, disease_type):
        """Create a 3D surface plot showing the relationship between three parameters."""
        # Get parameter data
        param_data = get_user_prediction_parameters(username, disease_type)
        
        if not param_data or len(param_data) == 0:
            st.info(f"No parameter data available for {self.disease_options[disease_type]}.")
            return
        
        # Extract parameters
        parameters = {}
        
        for record in param_data:
            if 'input_data' in record and 'timestamp' in record:
                timestamp = record['timestamp']
                
                for param, value in record['input_data'].items():
                    try:
                        # Try to convert to float
                        value = float(value)
                        if param not in parameters:
                            parameters[param] = []
                        parameters[param].append((timestamp, value))
                    except (ValueError, TypeError):
                        # Skip non-numeric parameters
                        continue
        
        if not parameters:
            st.info("No numeric parameters found for visualization.")
            return
        
        # Need at least 3 parameters for 3D visualization
        if len(parameters) < 3:
            st.info("Need at least 3 numeric parameters for 3D visualization.")
            return
        
        # For a surface plot, we need enough data points
        # If we don't have enough real data points, we can generate synthetic data
        if sum(len(v) for v in parameters.values()) / len(parameters) < 5:
            st.warning("Limited data available. Using synthetic data to enhance visualization.")
            use_synthetic = True
        else:
            use_synthetic = False
        
        # Allow user to select parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_param = st.selectbox("X-axis Parameter", list(parameters.keys()), index=0, key="surf_x")
        
        with col2:
            y_param = st.selectbox("Y-axis Parameter", list(parameters.keys()), 
                                index=min(1, len(parameters.keys())-1), key="surf_y")
        
        with col3:
            z_param = st.selectbox("Z-axis Parameter (Surface Height)", list(parameters.keys()), 
                                index=min(2, len(parameters.keys())-1), key="surf_z")
        
        # Create the visualization
        if x_param in parameters and y_param in parameters and z_param in parameters:
            # Extract parameter data
            x_data = [v for _, v in parameters[x_param]]
            y_data = [v for _, v in parameters[y_param]]
            z_data = [v for _, v in parameters[z_param]]
            
            # Find common timestamps
            common_data = []
            x_dict = dict(parameters[x_param])
            y_dict = dict(parameters[y_param])
            z_dict = dict(parameters[z_param])
            
            for ts in set(x_dict.keys()).intersection(set(y_dict.keys())).intersection(set(z_dict.keys())):
                common_data.append((x_dict[ts], y_dict[ts], z_dict[ts]))
            
            if len(common_data) >= 4:
                # Create grid for surface
                xi = np.linspace(min(x_data), max(x_data), 20)
                yi = np.linspace(min(y_data), max(y_data), 20)
                
                # Create mesh grid
                X, Y = np.meshgrid(xi, yi)
                
                # Grid data
                points = np.array([(x, y) for x, y, _ in common_data])
                values = np.array([z for _, _, z in common_data])
                
                # Interpolate
                Z = griddata(points, values, (X, Y), method='cubic')
                
                # Create 3D surface
                fig = go.Figure(data=[go.Surface(
                    x=xi,
                    y=yi,
                    z=Z,
                    colorscale='Viridis'
                )])
                
                # Add the original data points
                fig.add_trace(go.Scatter3d(
                    x=[x for x, _, _ in common_data],
                    y=[y for _, y, _ in common_data],
                    z=[z for _, _, z in common_data],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color='red',
                        opacity=0.8
                    ),
                    name='Data Points'
                ))
                
                fig.update_layout(
                    title=f'3D Surface Plot: {z_param} as a function of {x_param} and {y_param}',
                    scene=dict(
                        xaxis_title=x_param,
                        yaxis_title=y_param,
                        zaxis_title=z_param,
                        xaxis=dict(backgroundcolor="rgba(30, 30, 46, 0.8)"),
                        yaxis=dict(backgroundcolor="rgba(30, 30, 46, 0.8)"),
                        zaxis=dict(backgroundcolor="rgba(30, 30, 46, 0.8)")
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                    paper_bgcolor='rgba(30, 30, 46, 0.8)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough common data points to create a surface plot.")
        else:
            st.info("Please select valid parameters for visualization.")

    def create_3d_mesh_plot(self, username, disease_type):
        """Create a 3D mesh plot visualization."""
        # Get parameter data
        param_data = get_user_prediction_parameters(username, disease_type)
        
        if not param_data or len(param_data) == 0:
            st.info(f"No parameter data available for {self.disease_options[disease_type]}.")
            return
        
        # Extract parameters
        parameters = {}
        
        for record in param_data:
            if 'input_data' in record and 'timestamp' in record:
                timestamp = record['timestamp']
                
                for param, value in record['input_data'].items():
                    try:
                        # Try to convert to float
                        value = float(value)
                        if param not in parameters:
                            parameters[param] = []
                        parameters[param].append((timestamp, value))
                    except (ValueError, TypeError):
                        # Skip non-numeric parameters
                        continue
        
        if not parameters:
            st.info("No numeric parameters found for visualization.")
            return
        
        # Need at least 3 parameters for 3D visualization
        if len(parameters) < 3:
            st.info("Need at least 3 numeric parameters for 3D visualization.")
            return
        
        # Allow user to select parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_param = st.selectbox("X-axis Parameter", list(parameters.keys()), index=0, key="3d_mesh_x")
        
        with col2:
            y_param = st.selectbox("Y-axis Parameter", list(parameters.keys()), 
                                 index=min(1, len(parameters.keys())-1), key="3d_mesh_y")
        
        with col3:
            z_param = st.selectbox("Z-axis Parameter", list(parameters.keys()), 
                                 index=min(2, len(parameters.keys())-1), key="3d_mesh_z")
        
        # Create the visualization
        if x_param in parameters and y_param in parameters and z_param in parameters:
            # Extract parameter data
            x_data = [v for _, v in parameters[x_param]]
            y_data = [v for _, v in parameters[y_param]]
            z_data = [v for _, v in parameters[z_param]]
            
            # Find common timestamps to get matching data points
            common_data = []
            x_dict = dict(parameters[x_param])
            y_dict = dict(parameters[y_param])
            z_dict = dict(parameters[z_param])
            
            for ts in set(x_dict.keys()).intersection(set(y_dict.keys())).intersection(set(z_dict.keys())):
                common_data.append((x_dict[ts], y_dict[ts], z_dict[ts]))
            
            if len(common_data) < 4:
                st.info("Not enough common data points to create a mesh plot. Need at least 4 data points.")
                return
            
            # Create mesh grid using parameter data
            # Create grid for mesh
            x_min, x_max = min(x_data), max(x_data)
            y_min, y_max = min(y_data), max(y_data)
            
            # For mesh, we'll use a simpler grid
            xi = np.linspace(x_min, x_max, 10)
            yi = np.linspace(y_min, y_max, 10)
            
            # Create mesh grid
            X, Y = np.meshgrid(xi, yi)
            
            # Grid data
            points = np.array([(x, y) for x, y, _ in common_data])
            values = np.array([z for _, _, z in common_data])
            
            # Interpolate
            Z = griddata(points, values, (X, Y), method='linear')
            
            # Create 3D mesh plot
            fig = go.Figure()
            
            # Add mesh surface
            fig.add_trace(go.Mesh3d(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                opacity=0.8,
                colorscale='Viridis'
            ))
            
            # Add original data points
            fig.add_trace(go.Scatter3d(
                x=[x for x, _, _ in common_data],
                y=[y for _, y, _ in common_data],
                z=[z for _, _, z in common_data],
                mode='markers',
                marker=dict(
                    size=5,
                    color='red',
                    opacity=0.8
                ),
                name='Data Points'
            ))
            
            fig.update_layout(
                title=f'3D Mesh Plot: {z_param} as a function of {x_param} and {y_param}',
                scene=dict(
                    xaxis_title=x_param,
                    yaxis_title=y_param,
                    zaxis_title=z_param,
                    xaxis=dict(backgroundcolor="rgba(30, 30, 46, 0.8)"),
                    yaxis=dict(backgroundcolor="rgba(30, 30, 46, 0.8)"),
                    zaxis=dict(backgroundcolor="rgba(30, 30, 46, 0.8)")
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select valid parameters for visualization.") 