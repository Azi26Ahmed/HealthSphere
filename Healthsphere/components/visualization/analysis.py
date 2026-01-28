import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components.database import (
    get_user_predictions,
    get_disease_specific_stats,
    get_user_prediction_parameters
)
from .utils import format_disease_name, get_color_palette

class AnalysisVisualizer:
    """
    Analysis visualization class for health data analysis.
    Contains methods for trend analysis, correlation analysis, and health assessments.
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

    def create_overall_health_plot(self, username, disease_type=None):
        """Create a pie chart showing health assessment distribution by disease type."""
        # Get prediction data from database
        prediction_data = get_user_predictions(username, disease_type)
        
        if not prediction_data or "predictions" not in prediction_data or not prediction_data["predictions"]:
            st.info("No prediction data available for visualization.")
            return
            
        # Extract predictions
        predictions = prediction_data["predictions"]
        
        # Count predictions by disease type and risk level
        disease_counts = {}
        
        for prediction in predictions:
            if 'disease_type' in prediction and 'prediction_result' in prediction:
                d_type = prediction['disease_type']
                is_high_risk = prediction['prediction_result']
                risk = 'High Risk' if is_high_risk else 'Low Risk'
                
                if d_type not in disease_counts:
                    disease_counts[d_type] = {'Low Risk': 0, 'High Risk': 0}
                
                disease_counts[d_type][risk] += 1
        
        if not disease_counts:
            st.info("No valid prediction data found for visualization.")
            return
            
        # Prepare data for visualization
        plot_data = []
        
        for d_type, risks in disease_counts.items():
            disease_name = self.disease_options.get(d_type, d_type.replace('_', ' ').title())
            
            for risk, count in risks.items():
                if count > 0:
                    plot_data.append({
                        'Disease': disease_name,
                        'Risk Level': risk,
                        'Count': count
                    })
        
        if not plot_data:
            st.info("No plot data could be generated from predictions.")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(plot_data)
        
        # Create color map for risk levels
        color_map = {
            'Low Risk': '#4CAF50',      # Green
            'High Risk': '#F44336'      # Red
        }
        
        # Create pie chart for each disease type
        st.subheader("Health Assessment Distribution")
        
        # Group by disease
        disease_groups = df.groupby('Disease')
        
        for disease, group in disease_groups:
            risk_data = group.groupby('Risk Level')['Count'].sum().reset_index()
            
            # Create pie chart
            fig = px.pie(
                risk_data,
                values='Count',
                names='Risk Level',
                title=f'Risk Assessment for {disease}',
                color='Risk Level',
                color_discrete_map=color_map
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(30, 30, 46, 0.8)',
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Create overall summary if multiple diseases
        if len(disease_groups) > 1:
            overall_counts = df.groupby('Risk Level')['Count'].sum().reset_index()
            
            fig = px.pie(
                overall_counts,
                values='Count',
                names='Risk Level',
                title='Overall Health Risk Assessment',
                color='Risk Level',
                color_discrete_map=color_map
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(30, 30, 46, 0.8)',
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def create_risk_distribution_plot(self, username, disease_type=None):
        """Create a visualization showing risk distribution across diseases."""
        # Get prediction data from database
        prediction_data = get_user_predictions(username, disease_type)
        
        if not prediction_data or "predictions" not in prediction_data or not prediction_data["predictions"]:
            st.info("No prediction data available. Make some predictions first to see visualizations.")
            return
        
        # Convert to DataFrame
        predictions = prediction_data["predictions"]
        
        # Create a DataFrame with the prediction results
        df = pd.DataFrame([
            {
                'date': p['timestamp'],
                'disease_type': p.get('disease_type', 'unknown'),
                'result': 'High Risk' if p.get('prediction_result', False) else 'Low Risk',
                'confidence': p.get('confidence', 0),
                'is_high_risk': p.get('prediction_result', False)
            } for p in predictions
        ])
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Display distribution visualization
        st.subheader("Risk Distribution Analysis")
        
        # Group by disease type
        disease_counts = df.groupby('disease_type')['result'].count().reset_index()
        disease_counts.columns = ['Disease Type', 'Count']
        
        # Replace disease keys with human-readable names
        disease_counts['Disease Type'] = disease_counts['Disease Type'].map(
            lambda x: self.disease_options.get(x, x.replace('_', ' ').title())
        )
        
        # Count risk levels by disease type
        risk_counts = df.groupby(['disease_type', 'result']).size().reset_index(name='Count')
        risk_counts['disease_type'] = risk_counts['disease_type'].map(
            lambda x: self.disease_options.get(x, x.replace('_', ' ').title())
        )
        
        # Create enhanced distribution visualization
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Pie chart for overall disease distribution
            if len(disease_counts) > 0:
                fig = px.pie(
                    disease_counts,
                    values='Count',
                    names='Disease Type',
                    title='Prediction Distribution by Disease Type',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hoverinfo='label+percent+value',
                    marker=dict(line=dict(color='white', width=2))
                )
                
                fig.update_layout(
                    showlegend=False,
                    margin=dict(l=20, r=20, t=60, b=20),
                    plot_bgcolor='rgba(30, 30, 46, 0.8)',
                    paper_bgcolor='rgba(30, 30, 46, 0.8)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Stacked bar chart for risk levels by disease type
            if len(risk_counts) > 0:
                fig = px.bar(
                    risk_counts,
                    x='disease_type',
                    y='Count',
                    color='result',
                    title='Risk Distribution by Disease Type',
                    color_discrete_map={'High Risk': '#ff7f0e', 'Low Risk': '#1f77b4'},
                    barmode='stack'
                )
                
                fig.update_layout(
                    xaxis_title='Disease Type',
                    yaxis_title='Number of Predictions',
                    legend_title='Risk Level',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=20, r=20, t=60, b=20),
                    plot_bgcolor='rgba(30, 30, 46, 0.8)',
                    paper_bgcolor='rgba(30, 30, 46, 0.8)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)

    def create_prediction_history(self, username, disease_type):
        """Create a visualization of prediction history."""
        # Get prediction data from database
        prediction_data = get_user_predictions(username, disease_type)
        
        if not prediction_data or "predictions" not in prediction_data or not prediction_data["predictions"]:
            st.info(f"No {self.disease_options[disease_type]} prediction data available.")
            return
        
        # Convert to DataFrame
        predictions = prediction_data["predictions"]
        if len(predictions) == 0:
            st.info(f"No prediction data available for {self.disease_options[disease_type]}.")
            return
            
        # Create a DataFrame with the prediction results
        df = pd.DataFrame([
            {
                'date': p['timestamp'],
                'result': 'High Risk' if p.get('prediction_result', False) else 'Low Risk',
                'confidence': p.get('confidence', 0),
                'result_value': 1 if p.get('prediction_result', False) else 0  # Numeric value for scatter
            } for p in predictions
        ])
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create a figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add scatter points for high risk predictions
        high_risk_df = df[df['result'] == 'High Risk']
        low_risk_df = df[df['result'] == 'Low Risk']
        
        # Add scatter plot for predictions, with different colors/symbols for risk levels
        fig.add_trace(
            go.Scatter(
                x=high_risk_df['date'],
                y=high_risk_df['confidence'],
                mode='markers',
                name='High Risk',
                marker=dict(
                    size=12,
                    color='#ff0000',
                    symbol='circle',
                    line=dict(width=2, color='#ff5000')
                ),
                hovertemplate='<b>High Risk</b><br>Date: %{x}<br>Confidence: %{y:.1%}<extra></extra>'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=low_risk_df['date'],
                y=low_risk_df['confidence'],
                mode='markers',
                name='Low Risk',
                marker=dict(
                    size=12,
                    color='#00a0ff',
                    symbol='circle',
                    line=dict(width=2, color='#005fff')
                ),
                hovertemplate='<b>Low Risk</b><br>Date: %{x}<br>Confidence: %{y:.1%}<extra></extra>'
            )
        )
        
        # Add a line connecting all points chronologically
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['confidence'],
                mode='lines',
                line=dict(color='rgba(100, 100, 100, 0.3)', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        
        # Enhance layout
        fig.update_layout(
            title=f"Prediction History for {self.disease_options[disease_type]}",
            xaxis=dict(
                title="Date",
                gridcolor="rgba(100, 100, 100, 0.2)",
                showgrid=True
            ),
            yaxis=dict(
                title="Confidence Score",
                gridcolor="rgba(100, 100, 100, 0.2)",
                tickformat=".0%",
                showgrid=True
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode="x unified",
            plot_bgcolor='rgba(30, 30, 46, 0.8)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display a summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total Predictions",
                len(df),
                delta=None
            )
        with col2:
            high_risk_count = len(high_risk_df)
            if len(df) > 0:
                high_risk_percentage = f"{high_risk_count / len(df) * 100:.1f}%"
            else:
                high_risk_percentage = "N/A"
            st.metric(
                "High Risk Predictions",
                high_risk_count,
                delta=high_risk_percentage
            )

    def create_parameter_trends(self, username, disease_type):
        """Create parameter trend visualizations."""
        # Get parameter data from database
        param_data = get_user_prediction_parameters(username, disease_type)
        
        if not param_data or len(param_data) == 0:
            st.info(f"No parameter data available for {self.disease_options[disease_type]}.")
            return
        
        # Extract parameters
        parameters = {}
        dates = []
        
        for record in param_data:
            if 'input_data' in record and 'timestamp' in record:
                date = record['timestamp']
                dates.append(date)
                
                for param, value in record['input_data'].items():
                    try:
                        # Try to convert value to float
                        value = float(value)
                        if param not in parameters:
                            parameters[param] = []
                        parameters[param].append((date, value))
                    except (ValueError, TypeError):
                        # Skip non-numeric parameters
                        continue
        
        if not parameters:
            st.info("No numeric parameters found for trend analysis.")
            return
        
        # Allow user to select parameters to visualize
        selected_params = st.multiselect(
            "Select Parameters to Visualize", 
            list(parameters.keys()),
            default=[list(parameters.keys())[0]] if parameters else []
        )
        
        if not selected_params:
            st.info("Please select at least one parameter to visualize.")
            return
        
        # Create trend visualization
        st.subheader("Parameter Trends Over Time")
        
        # Prepare data for visualization
        trend_data = []
        
        for param in selected_params:
            if param in parameters:
                param_values = parameters[param]
                # Sort by date
                param_values.sort(key=lambda x: x[0])
                
                for date, value in param_values:
                    trend_data.append({
                        'Parameter': param,
                        'Date': date,
                        'Value': value
                    })
        
        if not trend_data:
            st.info("No trend data available for the selected parameters.")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(trend_data)
        
        # Create trend visualization - adapt based on number of data points
        if len(df) <= 3:
            # For few data points, use a bar chart
            fig = px.bar(
                df,
                x='Date',
                y='Value',
                color='Parameter',
                title=f'Parameter Trends for {self.disease_options[disease_type]}',
                barmode='group'
            )
        else:
            # For more data points, use a line chart
            fig = px.line(
                df,
                x='Date',
                y='Value',
                color='Parameter',
                title=f'Parameter Trends for {self.disease_options[disease_type]}',
                markers=True
            )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Parameter Value',
            legend_title='Parameter',
            plot_bgcolor='rgba(30, 30, 46, 0.8)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def create_trend_analysis(self, username, disease_type):
        """Create a trend analysis visualization showing changes over time."""
        # Get disease-specific stats
        stats = get_disease_specific_stats(username, disease_type)
        
        if not stats:
            st.info(f"No {self.disease_options[disease_type]} prediction data available for trend analysis.")
            return
        
        # Also get prediction data to show time-based trends
        prediction_data = get_user_predictions(username, disease_type)
        has_timeline_data = (prediction_data and "predictions" in prediction_data 
                            and len(prediction_data["predictions"]) > 0)
        
        # Display basic statistics with attractive metrics
        st.subheader("Trend Analysis Dashboard")
        
        # Row 1: Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style="background-color:#1f3347;padding:15px;border-radius:10px;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.3);border:1px solid #2c4a6a;">
                <h3 style="color:#ffffff;margin-bottom:5px;font-weight:600;">Total Predictions</h3>
                <h1 style="color:#4fa3fd;font-size:2.2em;margin:5px 0;">{}</h1>
            </div>
            """.format(stats.get('total_predictions', 0)), unsafe_allow_html=True)
        
        with col2:
            positive_pct = stats.get('positive_predictions', 0) / stats.get('total_predictions', 1) * 100
            st.markdown("""
            <div style="background-color:#3d2e1b;padding:15px;border-radius:10px;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.3);border:1px solid #5a4520;">
                <h3 style="color:#ffffff;margin-bottom:5px;font-weight:600;">High Risk Count</h3>
                <h1 style="color:#ffa726;font-size:2.2em;margin:5px 0;">{} <span style="font-size:16px;color:#dddddd;">({:.1f}%)</span></h1>
            </div>
            """.format(stats.get('positive_predictions', 0), positive_pct), unsafe_allow_html=True)
        
        with col3:
            avg_confidence = stats.get('average_confidence', 0) 
            # Ensure avg_confidence is not None before multiplying
            if avg_confidence is None:
                avg_confidence = 0
            avg_confidence_pct = avg_confidence * 100
            st.markdown(f"""
            <div style="background-color:#1b3321;padding:15px;border-radius:10px;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.3);border:1px solid #2b5233;">
                <h3 style="color:#ffffff;margin-bottom:5px;font-weight:600;">Avg Confidence</h3>
                <h1 style="color:#4caf50;font-size:2.2em;margin:5px 0;">{avg_confidence_pct:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Add timeline visualization if data is available
        if has_timeline_data:
            st.subheader("Prediction Trends Over Time")
            
            # Create DataFrame with time-based prediction data
            timeline_df = pd.DataFrame([
                {
                    'date': p['timestamp'],
                    'result': 'High Risk' if p.get('prediction_result', False) else 'Low Risk',
                    'confidence': p.get('confidence', 0),
                    'is_high_risk': p.get('prediction_result', False)
                } for p in prediction_data["predictions"]
            ])
            
            # Sort by date
            timeline_df['date'] = pd.to_datetime(timeline_df['date'])
            timeline_df = timeline_df.sort_values('date')
            
            # Group by month to show trends
            timeline_df['month'] = timeline_df['date'].dt.to_period('M').astype(str)
            monthly_count = timeline_df.groupby('month').size().reset_index(name='count')
            monthly_high_risk = timeline_df[timeline_df['is_high_risk']].groupby('month').size().reset_index(name='high_risk_count')
            
            # Merge the monthly data
            monthly_data = pd.merge(monthly_count, monthly_high_risk, on='month', how='left')
            monthly_data['high_risk_count'] = monthly_data['high_risk_count'].fillna(0)
            monthly_data['low_risk_count'] = monthly_data['count'] - monthly_data['high_risk_count']
            monthly_data['high_risk_pct'] = monthly_data['high_risk_count'] / monthly_data['count'] * 100
            
            # Create month-by-month trend visualization
            fig = go.Figure()
            
            # Add bars for prediction counts
            fig.add_trace(
                go.Bar(
                    x=monthly_data['month'],
                    y=monthly_data['high_risk_count'],
                    name='High Risk',
                    marker_color='rgba(255, 127, 14, 0.7)'
                )
            )
            
            fig.add_trace(
                go.Bar(
                    x=monthly_data['month'],
                    y=monthly_data['low_risk_count'],
                    name='Low Risk',
                    marker_color='rgba(31, 119, 180, 0.7)'
                )
            )
            
            # Add line for high risk percentage
            fig.add_trace(
                go.Scatter(
                    x=monthly_data['month'],
                    y=monthly_data['high_risk_pct'],
                    mode='lines+markers',
                    name='High Risk %',
                    yaxis='y2',
                    line=dict(color='red', width=3),
                    marker=dict(size=8)
                )
            )
            
            # Update layout for dual axes
            fig.update_layout(
                title='Monthly Prediction Trends',
                xaxis=dict(title='Month'),
                yaxis=dict(
                    title='Number of Predictions',
                    showgrid=True,
                    gridcolor='rgba(211, 211, 211, 0.3)'
                ),
                yaxis2=dict(
                    title='High Risk %',
                    overlaying='y',
                    side='right',
                    showgrid=False,
                    range=[0, 100],
                    ticksuffix='%'
                ),
                barmode='stack',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=20, r=60, t=40, b=20),
                plot_bgcolor='rgba(30, 30, 46, 0.8)',
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def create_feature_comparison(self, username, disease_type):
        """Create a feature comparison visualization."""
        # Get parameter data
        param_data = get_user_prediction_parameters(username, disease_type)
        
        if not param_data or len(param_data) == 0:
            st.info(f"No parameter data available for {self.disease_options[disease_type]}.")
            return
        
        # Extract parameters from the most recent prediction
        recent_params = {}
        if param_data:
            # Sort by timestamp to get the most recent
            param_data.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            if 'input_data' in param_data[0]:
                recent_params = param_data[0]['input_data']
        
        # Compare with averages if we have multiple records
        if len(param_data) > 1:
            # Calculate averages
            param_avgs = {}
            param_counts = {}
            
            for record in param_data:
                if 'input_data' in record:
                    for param, value in record['input_data'].items():
                        try:
                            value = float(value)
                            if param not in param_avgs:
                                param_avgs[param] = 0
                                param_counts[param] = 0
                            param_avgs[param] += value
                            param_counts[param] += 1
                        except (ValueError, TypeError):
                            # Skip non-numeric parameters
                            continue
            
            # Calculate averages
            for param in param_avgs:
                if param_counts[param] > 0:
                    param_avgs[param] /= param_counts[param]
            
            # Display comparison
            st.subheader("Feature Comparison: Most Recent vs. Average")
            
            # Prepare data for visualization
            comparison_data = []
            
            for param, avg_value in param_avgs.items():
                if param in recent_params:
                    try:
                        recent_value = float(recent_params[param])
                        comparison_data.append({
                            'Parameter': param,
                            'Most Recent': recent_value,
                            'Average': avg_value
                        })
                    except (ValueError, TypeError):
                        continue
        
            if comparison_data:
                # Convert to DataFrame for visualization
                df = pd.DataFrame(comparison_data)
                
                # Create a bar chart for the comparison
                fig = go.Figure()
                
                for i, row in df.iterrows():
                    fig.add_trace(go.Bar(
                        name='Most Recent',
                        x=[row['Parameter']],
                        y=[row['Most Recent']],
                        marker_color=self.color_themes['primary'][0]
                    ))
                    fig.add_trace(go.Bar(
                        name='Average',
                        x=[row['Parameter']],
                        y=[row['Average']],
                        marker_color=self.color_themes['primary'][1]
                    ))
                
                fig.update_layout(
                    title=f'Most Recent vs. Average Parameters for {self.disease_options[disease_type]}',
                    yaxis_title='Value',
                    barmode='group',
                    legend_title='Type',
                    plot_bgcolor='rgba(30, 30, 46, 0.8)',
                    paper_bgcolor='rgba(30, 30, 46, 0.8)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No comparable parameters available.")
        else:
            # If only one record, just display the parameters
            st.subheader("Feature Values")
            
            # Display as a table
            if recent_params:
                param_df = pd.DataFrame([
                    {'Parameter': param, 'Value': value}
                    for param, value in recent_params.items()
                ])
                st.dataframe(param_df)

    def create_parameter_correlation_plot(self, username, disease_type):
        """Create a correlation plot between different parameters."""
        # Get parameter data
        param_data = get_user_prediction_parameters(username, disease_type)
        
        if not param_data or len(param_data) == 0:
            st.info(f"No {self.disease_options[disease_type]} prediction data available.")
            return
        
        # Extract parameters from all predictions
        all_params = {}
        timestamps = []
        
        for record in param_data:
            if 'input_data' in record and 'timestamp' in record:
                timestamp = record['timestamp']
                timestamps.append(timestamp)
                
                for param, value in record['input_data'].items():
                    try:
                        # Try to convert value to float
                        value = float(value)
                        if param not in all_params:
                            all_params[param] = []
                        all_params[param].append((timestamp, value))
                    except (ValueError, TypeError):
                        # Skip non-numeric parameters
                        continue
        
        # Need at least 2 parameters with data for correlation
        numeric_params = list(all_params.keys())
        if len(numeric_params) < 2:
            st.info("Not enough numeric parameters found for correlation analysis.")
            self.show_synthetic_correlation(disease_type)
            return
        
        # Handle case with too few data points
        if len(timestamps) < 2:
            st.warning("Not enough data points for meaningful correlation analysis.")
            self.show_synthetic_correlation(disease_type)
            return
            
        # Function to create correlation matrix from available data
        def create_correlation_matrix(param_data, timestamps):
            # Create a DataFrame with all parameter values
            data_points = {}
            
            for param, values in param_data.items():
                value_dict = dict(values)
                data_points[param] = [value_dict.get(ts, np.nan) for ts in timestamps]
            
            df = pd.DataFrame(data_points)
            
            # Handle NaN values
            df = df.fillna(df.mean())
            
            # Check if we have enough valid data points
            if df.shape[0] < 2 or df.isnull().any().any():
                # If we have NaN values after trying to fill them, data is insufficient
                return None
                
            # Calculate correlation matrix
            try:
                corr_matrix = df.corr(method='pearson', min_periods=1)
                # Check for NaN values in correlation matrix
                if corr_matrix.isnull().any().any():
                    return None
                return corr_matrix
            except Exception:
                return None
        
        # Calculate correlation matrix
        correlation_matrix = create_correlation_matrix(all_params, timestamps)
        
        if correlation_matrix is None:
            st.warning("Unable to calculate correlations due to insufficient data.")
            self.show_synthetic_correlation(disease_type)
            return
        
        # Display heatmap of correlations
        st.subheader("Parameter Correlation Analysis")
        
        # Create an enhanced heatmap
        fig = go.Figure()
        
        # Add heatmap for correlation matrix
        heatmap = go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=[[f"{val:.2f}" for val in row] for row in correlation_matrix.values],
            hovertemplate='%{y} & %{x}<br>Correlation: %{z:.3f}<extra></extra>',
            showscale=True,
            colorbar=dict(
                title=dict(text='Correlation'),
                thickness=15,
                thicknessmode='pixels',
                len=0.9,
                lenmode='fraction',
                outlinewidth=0
            )
        )
        
        fig.add_trace(heatmap)
        
        # Add text annotations with correlation values
        for i, row in enumerate(correlation_matrix.index):
            for j, col in enumerate(correlation_matrix.columns):
                value = correlation_matrix.iloc[i, j]
                text_color = 'white' if abs(value) > 0.4 else 'black'
                fig.add_annotation(
                    x=col,
                    y=row,
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(
                        color=text_color,
                        size=10
                    )
                )
                
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=f"Parameter Correlation Matrix for {self.disease_options[disease_type]}",
                font=dict(size=18)
            ),
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor='rgba(30, 30, 46, 0.8)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font_color='white',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def show_synthetic_correlation(self, disease_type):
        """Show a synthetic correlation matrix when real data is insufficient."""
        st.info("Showing a synthetic example of parameter correlations.")
        
        # Create synthetic correlation data
        np.random.seed(42)
        if disease_type == 'heart':
            params = ["Age", "Blood Pressure", "Cholesterol", "Max Heart Rate", "ST Depression"]
        elif disease_type == 'diabetes':
            params = ["Glucose", "Blood Pressure", "BMI", "Age", "Insulin"]
        else:
            params = ["Parameter 1", "Parameter 2", "Parameter 3", "Parameter 4", "Parameter 5"]
            
        n_params = len(params)
        
        # Generate a realistic correlation matrix (symmetric, diagonal=1)
        corr_matrix = np.random.uniform(-0.8, 0.8, size=(n_params, n_params))
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make it symmetric
        np.fill_diagonal(corr_matrix, 1)  # Fill diagonal with 1s
        
        # Convert to DataFrame
        corr_df = pd.DataFrame(corr_matrix, index=params, columns=params)
        
        # Create heatmap
        fig = go.Figure()
        
        # Add heatmap
        heatmap = go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale='RdBu_r',
            zmid=0,
            text=[[f"{val:.2f}" for val in row] for row in corr_df.values],
            hovertemplate='%{y} & %{x}<br>Correlation: %{z:.3f}<extra></extra>',
            showscale=True,
            colorbar=dict(
                title=dict(text='Correlation'),
                thickness=15,
                thicknessmode='pixels',
                len=0.9,
                lenmode='fraction',
                outlinewidth=0
            )
        )
        
        fig.add_trace(heatmap)
        
        # Add text annotations
        for i, row in enumerate(corr_df.index):
            for j, col in enumerate(corr_df.columns):
                value = corr_df.iloc[i, j]
                text_color = 'white' if abs(value) > 0.4 else 'black'
                fig.add_annotation(
                    x=col,
                    y=row,
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(
                        color=text_color,
                        size=10
                    )
                )
                
        # Layout
        fig.update_layout(
            title=dict(
                text=f"Example Correlation Matrix for {self.disease_options[disease_type]} (Synthetic Data)",
                font=dict(size=18)
            ),
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor='rgba(30, 30, 46, 0.8)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font_color='white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info("This is a synthetic example. Make more predictions to see actual correlations in your data.")

    def create_disease_risk_timeline(self, username, disease_type):
        """Create a timeline visualization of disease risk predictions."""
        # Get prediction data from database
        prediction_data = get_user_predictions(username, disease_type)
        
        if not prediction_data or "predictions" not in prediction_data or not prediction_data["predictions"]:
            st.info(f"No {self.disease_options[disease_type]} prediction data available.")
            return
        
        # Convert to DataFrame
        predictions = prediction_data["predictions"]
        if len(predictions) == 0:
            st.info(f"No prediction data available for {self.disease_options[disease_type]}.")
            return
            
        # Create a DataFrame with the prediction results
        df = pd.DataFrame([
            {
                'date': p['timestamp'],
                'result': 'High Risk' if p.get('prediction_result', False) else 'Low Risk',
                'confidence': p.get('confidence', 0),
                'risk_score': float(p.get('confidence', 0.5)) if p.get('prediction_result', False) else 1 - float(p.get('confidence', 0.5)),
                'is_high_risk': p.get('prediction_result', False)
            } for p in predictions
        ])
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Display timeline visualization
        st.subheader(f"Risk Timeline for {self.disease_options[disease_type]}")
        
        # Create an enhanced timeline visualization
        fig = go.Figure()
        
        # Add risk score line
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['risk_score'],
                mode='lines+markers',
                name='Risk Score',
                line=dict(
                    color='rgba(255, 127, 14, 0.7)',
                    width=3
                ),
                marker=dict(
                    size=10,
                    color=df['is_high_risk'].map({True: 'red', False: 'green'}),
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Risk Score: %{y:.2f}<extra></extra>',
                text=df['result']
            )
        )
        
        # Add threshold line for high risk
        fig.add_shape(
            type="line",
            x0=df['date'].min(),
            y0=0.5,  # Assuming 0.5 is the threshold
            x1=df['date'].max(),
            y1=0.5,
            line=dict(color="red", width=2, dash="dash"),
            name="Risk Threshold"
        )
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=f'Disease Risk Timeline for {self.disease_options[disease_type]}',
                font=dict(size=22)
            ),
            xaxis=dict(
                title='Date',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(211, 211, 211, 0.3)'
            ),
            yaxis=dict(
                title='Risk Score (0-1)',
                range=[0, 1],
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(211, 211, 211, 0.3)'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='closest',
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor='rgba(30, 30, 46, 0.8)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add risk trends summary
        if len(df) >= 2:
            # Calculate trend metrics
            recent_predictions = df.tail(min(5, len(df)))
            recent_high_risk = recent_predictions['is_high_risk'].mean() * 100
            all_high_risk = df['is_high_risk'].mean() * 100
            trend_direction = recent_high_risk - all_high_risk
            
            # Display trend summary
            st.subheader("Risk Trend Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Predictions",
                    len(df),
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Overall High Risk %",
                    f"{all_high_risk:.1f}%",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Recent High Risk %",
                    f"{recent_high_risk:.1f}%",
                    delta=f"{trend_direction:+.1f}%" if abs(trend_direction) > 0.1 else None,
                    delta_color="inverse"  # Red for increasing risk (bad)
                ) 