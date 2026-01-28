import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from components.database import get_user_prediction_parameters, get_user_predictions

class Plots2DVisualizer:
    """
    2D visualization class for health data plotting.
    Contains methods for 2D line plots, scatter plots, and parameter visualizations.
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
        self.color_themes = {
            "primary": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
            "pastel": ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462"],
            "diverging": ["#d53e4f", "#fc8d59", "#fee08b", "#e6f598", "#99d594", "#3288bd"]
        }

    def create_2d_line_plot(self, username, disease_type):
        """Create a customizable 2D line plot."""
        # Get parameter data
        param_data = get_user_prediction_parameters(username, disease_type)
        
        if not param_data or len(param_data) == 0:
            st.info(f"No parameter data available for {self.disease_options[disease_type]}.")
            return
        
        # Extract parameters and timestamps
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
        
        # Allow user to select parameters for visualization
        col1, col2 = st.columns(2)
        
        with col1:
            x_options = ['Timestamp'] + list(parameters.keys())
            x_param = st.selectbox("X-axis Parameter", x_options, index=0)
        
        with col2:
            y_param = st.selectbox("Y-axis Parameter", list(parameters.keys()), index=0)
        
        # Create the visualization
        if x_param == 'Timestamp' and y_param in parameters:
            # Time series plot
            plot_data = []
            
            for timestamp, value in sorted(parameters[y_param], key=lambda x: x[0]):
                plot_data.append({
                    'Timestamp': timestamp,
                    'Value': value,
                    'Parameter': y_param
                })
            
            if not plot_data:
                st.info(f"No data available for {y_param}.")
                return
                
            df = pd.DataFrame(plot_data)
            
            # Create line chart, adapting based on number of data points
            if len(df) <= 3:
                # For few data points, use a scatter plot with lines
                fig = px.scatter(
                    df,
                    x='Timestamp',
                    y='Value',
                    title=f'{y_param} Over Time for {self.disease_options[disease_type]}',
                    color_discrete_sequence=[self.color_themes['primary'][0]]
                )
                
                # Add connecting lines if we have at least 2 points
                if len(df) >= 2:
                    fig.add_trace(
                        go.Scatter(
                            x=df['Timestamp'],
                            y=df['Value'],
                            mode='lines',
                            line=dict(color=self.color_themes['primary'][0]),
                            showlegend=False
                        )
                    )
            else:
                # With more data points, use a regular line chart
                fig = px.line(
                    df,
                    x='Timestamp',
                    y='Value',
                    title=f'{y_param} Over Time for {self.disease_options[disease_type]}',
                    markers=True
                )
            
            fig.update_layout(
                xaxis_title='Time',
                yaxis_title=y_param,
                plot_bgcolor='rgba(30, 30, 46, 0.8)',
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif x_param in parameters and y_param in parameters:
            # Parameter vs parameter plot
            x_values = dict(parameters[x_param])
            y_values = dict(parameters[y_param])
            
            # Find common timestamps
            common_timestamps = set(x_values.keys()).intersection(set(y_values.keys()))
            
            if not common_timestamps:
                st.info(f"No matching data points for {x_param} and {y_param}.")
                return
                
            # Create data for plotting
            plot_data = [
                {
                    'X Value': x_values[ts],
                    'Y Value': y_values[ts],
                    'Timestamp': ts
                }
                for ts in common_timestamps
            ]
            
            df = pd.DataFrame(plot_data)
            
            # Create scatter plot with option for trendline if enough points
            if len(df) <= 3:
                fig = px.scatter(
                    df,
                    x='X Value',
                    y='Y Value',
                    hover_data=['Timestamp'],
                    title=f'{y_param} vs {x_param} for {self.disease_options[disease_type]}',
                    color_discrete_sequence=[self.color_themes['primary'][0]]
                )
            else:
                fig = px.scatter(
                    df,
                    x='X Value',
                    y='Y Value',
                    hover_data=['Timestamp'],
                    title=f'{y_param} vs {x_param} for {self.disease_options[disease_type]}',
                    trendline="ols" if len(df) >= 5 else None
                )
            
            fig.update_layout(
                xaxis_title=x_param,
                yaxis_title=y_param,
                plot_bgcolor='rgba(30, 30, 46, 0.8)',
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select valid parameters for visualization.")

    def create_2d_scatter_plot(self, username, disease_type):
        """Create a customizable 2D scatter plot with additional options."""
        # Get parameter data
        param_data = get_user_prediction_parameters(username, disease_type)
        
        if not param_data or len(param_data) == 0:
            st.info(f"No parameter data available for {self.disease_options[disease_type]}.")
            return
        
        # Extract parameters and prediction results
        parameters = {}
        prediction_results = {}
        
        for i, record in enumerate(param_data):
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
        
        # Get prediction results
        prediction_data = get_user_predictions(username, disease_type)
        if prediction_data and "predictions" in prediction_data:
            for p in prediction_data["predictions"]:
                if 'timestamp' in p and 'prediction_result' in p:
                    prediction_results[p['timestamp']] = p['prediction_result']
        
        # Allow user to select parameters and visualization options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_param = st.selectbox("X-axis Parameter", list(parameters.keys()), index=0)
        
        with col2:
            y_param = st.selectbox("Y-axis Parameter", list(parameters.keys()), 
                                index=min(1, len(parameters.keys())-1))
        
        with col3:
            color_by = st.selectbox("Color by", ["Parameter Value", "Risk Level"], 
                                  disabled=(not prediction_results))
        
        # Create the visualization
        if x_param in parameters and y_param in parameters:
            # Get x and y values
            x_data = dict(parameters[x_param])
            y_data = dict(parameters[y_param])
            
            # Find common timestamps
            common_timestamps = set(x_data.keys()).intersection(set(y_data.keys()))
            
            if not common_timestamps:
                st.info(f"No matching data points for {x_param} and {y_param}.")
                return
                
            # Create data for plotting
            plot_data = []
            
            for ts in common_timestamps:
                data_point = {
                    'X Value': x_data[ts],
                    'Y Value': y_data[ts],
                    'Timestamp': ts
                }
                
                # Add prediction result if available and requested
                if color_by == "Risk Level" and ts in prediction_results:
                    data_point['Risk'] = 'High Risk' if prediction_results[ts] else 'Low Risk'
                else:
                    # Use a third parameter for color if available
                    for param in parameters:
                        if param != x_param and param != y_param:
                            param_data_dict = dict(parameters[param])
                            if ts in param_data_dict:
                                data_point['Color Value'] = param_data_dict[ts]
                                data_point['Color Parameter'] = param
                                break
                
                plot_data.append(data_point)
            
            df = pd.DataFrame(plot_data)
            
            # Create the scatter plot
            if color_by == "Risk Level" and 'Risk' in df.columns:
                fig = px.scatter(
                    df,
                    x='X Value',
                    y='Y Value',
                    color='Risk',
                    hover_data=['Timestamp'],
                    title=f'{y_param} vs {x_param} Colored by Risk Level',
                    color_discrete_map={'High Risk': '#ff7f0e', 'Low Risk': '#1f77b4'}
                )
            elif 'Color Value' in df.columns:
                fig = px.scatter(
                    df,
                    x='X Value',
                    y='Y Value',
                    color='Color Value',
                    hover_data=['Timestamp'],
                    title=f'{y_param} vs {x_param} Colored by {df["Color Parameter"].iloc[0]}',
                    color_continuous_scale='Viridis'
                )
            else:
                fig = px.scatter(
                    df,
                    x='X Value',
                    y='Y Value',
                    hover_data=['Timestamp'],
                    title=f'{y_param} vs {x_param} for {self.disease_options[disease_type]}'
                )
            
            # Add trendline if we have enough points
            if len(df) >= 5:
                x_vals = df['X Value']
                y_vals = df['Y Value']
                
                # Simple linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
                
                # Add trendline
                x_range = np.linspace(min(x_vals), max(x_vals), 100)
                y_range = intercept + slope * x_range
                
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_range,
                        mode='lines',
                        line=dict(color='rgba(255,255,255,0.5)'),
                        name=f'Trendline (r² = {r_value**2:.3f})'
                    )
                )
            
            fig.update_layout(
                xaxis_title=x_param,
                yaxis_title=y_param,
                plot_bgcolor='rgba(30, 30, 46, 0.8)',
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select valid parameters for visualization.")

    def create_2d_parameter_plot(self, username, disease_type):
        """Create a 2D parameter plot (wrapper for line plot)."""
        self.create_2d_line_plot(username, disease_type) 