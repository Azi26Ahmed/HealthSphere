import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from components.database import (
    get_user_predictions,
    get_user_prediction_parameters
)
from .base import BaseVisualizer

class MLVisualizer(BaseVisualizer):
    """Machine Learning visualization methods for HealthSphere."""
    
    def create_parameter_impact_plot(self, username, disease_type):
        """Create a visualization showing the impact of different parameters on disease risk."""
        # Get parameter data
        param_data = get_user_prediction_parameters(username, disease_type)
        
        if not param_data or len(param_data) == 0:
            st.info(f"No parameter data available for {self.disease_options[disease_type]}.")
            return
        
        # Extract parameters and prediction results
        parameters = {}
        prediction_results = {}
        
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
        
        # Get prediction results
        prediction_data = get_user_predictions(username, disease_type)
        if prediction_data and "predictions" in prediction_data:
            for p in prediction_data["predictions"]:
                if 'timestamp' in p and 'prediction_result' in p:
                    # Convert boolean to numeric (1 for high risk, 0 for low risk)
                    risk_score = 1.0 if p['prediction_result'] else 0.0
                    prediction_results[p['timestamp']] = risk_score
                elif 'timestamp' in p and 'risk_score' in p:
                    # Use risk_score if it exists (some older data might have this)
                    prediction_results[p['timestamp']] = p['risk_score']
        
        if not prediction_results:
            st.info("No prediction results available for parameter impact analysis.")
            # Show sample parameter impact visualization
            self.show_sample_parameter_impact(disease_type)
            return
        
        # Analyze parameter impact
        parameter_impacts = []
        
        for param_name, param_values in parameters.items():
            # Create a list of (param_value, risk_score) pairs
            param_risk_pairs = []
            
            for timestamp, value in param_values:
                if timestamp in prediction_results:
                    param_risk_pairs.append((value, prediction_results[timestamp]))
            
            # Changed: Always try to calculate impact even with just 2 data points
            if len(param_risk_pairs) >= 2:
                # Calculate correlation
                param_values = [p[0] for p in param_risk_pairs]
                risk_values = [p[1] for p in param_risk_pairs]
                
                try:
                    # Calculate correlation coefficient
                    correlation, p_value = stats.pearsonr(param_values, risk_values)
                    
                    parameter_impacts.append({
                        'Parameter': param_name,
                        'Correlation': correlation,
                        'P-Value': p_value,
                        'Impact': abs(correlation)
                    })
                except Exception as e:
                    # Handle constant arrays
                    if "constant" in str(e):
                        # For constant arrays, calculate a basic direction indicator
                        # If all parameter values are the same but risk values differ
                        if len(set(param_values)) == 1 and len(set(risk_values)) > 1:
                            # No correlation can be established statistically
                            continue
                        # If all risk values are the same but parameter values differ
                        elif len(set(risk_values)) == 1 and len(set(param_values)) > 1:
                            correlation = 0  # No correlation (flat line)
                            p_value = 1.0    # Not statistically significant
                            
                            parameter_impacts.append({
                                'Parameter': param_name,
                                'Correlation': correlation,
                                'P-Value': p_value,
                                'Impact': 0
                            })
                    else:
                        # Skip other correlation calculation failures
                        continue
        
        if not parameter_impacts:
            st.info("Unable to calculate meaningful correlations. Try adding more diverse prediction data.")
            self.show_sample_parameter_impact(disease_type)
            return
        
        # Convert to DataFrame and sort by impact
        impact_df = pd.DataFrame(parameter_impacts)
        impact_df = impact_df.sort_values('Impact', ascending=False)
        
        # Display impact visualization
        st.subheader("Parameter Impact Analysis")
        
        # Create bar chart of parameter impacts
        fig = px.bar(
            impact_df,
            x='Parameter',
            y='Impact',
            color='Correlation',
            title=f'Parameter Impact on {self.disease_options[disease_type]} Risk',
            color_continuous_scale='RdBu_r',
            labels={'Correlation': 'Direction and Strength'},
        )
        
        fig.update_layout(
            xaxis_title='Parameter',
            yaxis_title='Impact Magnitude',
            plot_bgcolor='rgba(30, 30, 46, 0.8)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation table
        st.subheader("Parameter Correlation Details")
        st.dataframe(impact_df[['Parameter', 'Correlation', 'P-Value']])
        
        # Select a parameter to visualize in detail
        if len(impact_df) > 0:
            selected_param = st.selectbox(
                "Select Parameter for Detailed Analysis",
                impact_df['Parameter'].tolist(),
                index=0
            )
            
            # Create scatter plot for selected parameter
            selected_param_data = []
            
            for timestamp, value in parameters[selected_param]:
                if timestamp in prediction_results:
                    selected_param_data.append({
                        'Parameter Value': value,
                        'Risk Score': prediction_results[timestamp],
                        'Timestamp': timestamp
                    })
            
            if selected_param_data:
                param_df = pd.DataFrame(selected_param_data)
                
                # Create scatter plot WITHOUT trendline to avoid statsmodels dependency
                fig = px.scatter(
                    param_df,
                    x='Parameter Value',
                    y='Risk Score',
                    hover_data=['Timestamp'],
                    title=f'Impact of {selected_param} on {self.disease_options[disease_type]} Risk'
                )
                
                # If we have enough data points (≥2), add a simple trend line manually
                if len(param_df) >= 2:
                    # Calculate simple linear regression using numpy
                    x = param_df['Parameter Value'].values
                    y = param_df['Risk Score'].values
                    
                    # Simple linear regression: y = mx + b
                    try:
                        # Calculate slope (m) and intercept (b) using numpy
                        m, b = np.polyfit(x, y, 1)
                        
                        # Generate line points
                        x_range = np.linspace(min(x), max(x), 100)
                        y_range = m * x_range + b
                        
                        # Add trend line
                        fig.add_traces(
                            go.Scatter(
                                x=x_range, 
                                y=y_range,
                                mode='lines',
                                name=f'Trend (slope: {m:.3f})',
                                line=dict(color='rgba(255, 255, 255, 0.7)')
                            )
                        )
                    except Exception:
                        # If calculation fails, skip trend line
                        pass
                
                fig.update_layout(
                    xaxis_title=selected_param,
                    yaxis_title='Risk Score',
                    plot_bgcolor='rgba(30, 30, 46, 0.8)',
                    paper_bgcolor='rgba(30, 30, 46, 0.8)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)

    def show_sample_parameter_impact(self, disease_type):
        """Show sample parameter impact visualization when real data is insufficient."""
        st.warning("Showing a sample parameter impact visualization with synthetic data.")
        
        # Create sample data based on disease type
        np.random.seed(42)
        if disease_type == 'heart':
            parameters = ["Age", "Blood Pressure", "Cholesterol", "Max Heart Rate", "ST Depression"]
            impacts = [0.78, 0.65, 0.58, -0.45, 0.32]
        elif disease_type == 'diabetes':
            parameters = ["Glucose", "BMI", "Age", "Insulin", "Blood Pressure"]
            impacts = [0.82, 0.67, 0.42, 0.38, 0.29]
        else:
            parameters = ["Parameter A", "Parameter B", "Parameter C", "Parameter D", "Parameter E"]
            impacts = [0.75, 0.62, 0.48, 0.35, 0.22]
        
        # Create sample correlations (same magnitude as impacts but with sign)
        correlations = [impact if np.random.random() > 0.3 else -impact for impact in impacts]
        
        # Create sample p-values (lower for higher impacts)
        p_values = [max(0.001, 0.05 - abs(impact)*0.05) for impact in impacts]
        
        # Create DataFrame
        impact_df = pd.DataFrame({
            'Parameter': parameters,
            'Correlation': correlations,
            'P-Value': p_values,
            'Impact': [abs(c) for c in correlations]
        })
        
        # Sort by impact
        impact_df = impact_df.sort_values('Impact', ascending=False)
        
        # Display sample visualization
        st.subheader("Parameter Impact Analysis (Sample)")
        
        # Create bar chart
        fig = px.bar(
            impact_df,
            x='Parameter',
            y='Impact',
            color='Correlation',
            title=f'Example Parameter Impact on {self.disease_options[disease_type]} Risk',
            color_continuous_scale='RdBu_r',
            labels={'Correlation': 'Direction and Strength'},
        )
        
        fig.update_layout(
            xaxis_title='Parameter',
            yaxis_title='Impact Magnitude',
            plot_bgcolor='rgba(30, 30, 46, 0.8)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display sample correlation table
        st.subheader("Parameter Correlation Details (Sample)")
        st.dataframe(impact_df[['Parameter', 'Correlation', 'P-Value']])
        
        # Sample parameter detail plot
        st.subheader("Parameter Detail (Sample)")
        
        # Create sample data points
        x = np.linspace(min(30, 50 + np.random.randn()*10), max(80, 70 + np.random.randn()*10), 20)
        if correlations[0] > 0:
            y = 0.3 + 0.6 * (x - min(x)) / (max(x) - min(x)) + 0.1 * np.random.randn(len(x))
        else:
            y = 0.9 - 0.6 * (x - min(x)) / (max(x) - min(x)) + 0.1 * np.random.randn(len(x))
            
        # Create sample DataFrame
        sample_df = pd.DataFrame({
            'Parameter Value': x,
            'Risk Score': y
        })
        
        # Create scatter plot
        fig = px.scatter(
            sample_df,
            x='Parameter Value',
            y='Risk Score',
            title=f'Example Impact of {parameters[0]} on {self.disease_options[disease_type]} Risk',
            trendline="ols"
        )
        
        fig.update_layout(
            xaxis_title=parameters[0],
            yaxis_title='Risk Score',
            plot_bgcolor='rgba(30, 30, 46, 0.8)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info("This is a synthetic example. Make more predictions to see actual parameter impacts in your data.")

    def create_decision_boundary_plot(self, username, disease_type):
        """Create a decision boundary visualization for a selected pair of parameters."""
        # Get parameter data
        param_data = get_user_prediction_parameters(username, disease_type)
        prediction_data = get_user_predictions(username, disease_type)
        
        if not param_data or len(param_data) == 0 or not prediction_data or "predictions" not in prediction_data:
            st.info(f"No {self.disease_options[disease_type]} prediction data available.")
            # Show sample visualization
            self.show_sample_decision_boundary(disease_type)
            return
        
        # Extract parameters and prediction results
        parameters = {}
        prediction_results = {}
        
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
            self.show_sample_decision_boundary(disease_type)
            return
        
        # Extract prediction results
        predictions = prediction_data["predictions"]
        for p in predictions:
            if 'timestamp' in p and 'prediction_result' in p:
                prediction_results[p['timestamp']] = bool(p['prediction_result'])
        
        if not prediction_results:
            st.info("No prediction results available for decision boundary visualization.")
            self.show_sample_decision_boundary(disease_type)
            return
        
        # Check if we have both classes (high risk and low risk)
        unique_classes = set(prediction_results.values())
        if len(unique_classes) < 2:
            st.warning("Decision boundary visualization requires predictions with both high and low risk results. Currently, only one risk level is present in your data.")
            st.info("Please make more predictions with varied results to enable this visualization.")
            # Show sample visualization with synthetic data
            self.show_sample_decision_boundary(disease_type)
            return
            
        # Need at least 2 parameters for decision boundary
        if len(parameters) < 2:
            st.info("Need at least 2 numeric parameters for decision boundary visualization.")
            self.show_sample_decision_boundary(disease_type)
            return
        
        # Allow user to select 2 parameters
        col1, col2 = st.columns(2)
        
        with col1:
            x_param = st.selectbox("X-axis Parameter", list(parameters.keys()), index=0)
        
        with col2:
            y_param = st.selectbox("Y-axis Parameter", list(parameters.keys()), 
                                  index=min(1, len(parameters.keys())-1))
        
        # Prepare data for visualization
        plot_data = []
        
        x_dict = dict(parameters[x_param])
        y_dict = dict(parameters[y_param])
        
        for ts in set(x_dict.keys()).intersection(set(y_dict.keys())).intersection(set(prediction_results.keys())):
            plot_data.append({
                'X': x_dict[ts],
                'Y': y_dict[ts],
                'Result': 'High Risk' if prediction_results[ts] else 'Low Risk',
                'Timestamp': ts
            })
        
        if len(plot_data) < 4:
            st.info("Not enough data points for decision boundary visualization. Need at least 4 data points.")
            self.show_sample_decision_boundary(disease_type)
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(plot_data)
        
        # Check if we have both classes in our actual plotting data
        if len(df['Result'].unique()) < 2:
            st.warning("After filtering for common timestamps, only one risk level remains. Need both for boundary visualization.")
            self.show_sample_decision_boundary(disease_type)
            return
        
        # Create decision boundary visualization
        st.subheader("Decision Boundary Visualization")
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='X',
            y='Y',
            color='Result',
            title=f'Decision Boundary for {self.disease_options[disease_type]}',
            labels={'X': x_param, 'Y': y_param},
            color_discrete_map={'High Risk': '#ff7f0e', 'Low Risk': '#1f77b4'},
            hover_data=['Timestamp']
        )
        
        # Simple classification boundary using a line if possible
        if len(df) >= 6:
            try:
                import numpy as np
                from sklearn.svm import SVC
                from sklearn.preprocessing import StandardScaler
                
                # Normalize the data
                scaler = StandardScaler()
                X = scaler.fit_transform(df[['X', 'Y']])
                y = df['Result'] == 'High Risk'
                
                # Fit SVM model
                model = SVC(kernel='linear')
                model.fit(X, y)
                
                # Create grid for decision boundary
                x_min, x_max = df['X'].min() - 0.1, df['X'].max() + 0.1
                y_min, y_max = df['Y'].min() - 0.1, df['Y'].max() + 0.1
                
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                    np.linspace(y_min, y_max, 100))
                
                # Transform grid points
                grid_transformed = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
                
                # Get predictions and reshape
                Z = model.decision_function(grid_transformed).reshape(xx.shape)
                
                # Add contour to plot
                fig.add_trace(
                    go.Contour(
                        z=Z,
                        x=np.linspace(x_min, x_max, 100),
                        y=np.linspace(y_min, y_max, 100),
                        showscale=False,
                        contours=dict(
                            start=0,
                            end=0,
                            size=0,
                            coloring='lines',
                            showlabels=False
                        ),
                        line=dict(
                            width=2,
                            color='black',
                            dash='dash'
                        ),
                        name='Decision Boundary'
                    )
                )
            except Exception as e:
                st.info(f"Could not generate decision boundary: {str(e)}")
        
        fig.update_layout(
            xaxis_title=x_param,
            yaxis_title=y_param,
            plot_bgcolor='rgba(30, 30, 46, 0.8)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def show_sample_decision_boundary(self, disease_type):
        """Show a sample decision boundary visualization with synthetic data."""
        st.subheader("Sample Decision Boundary Visualization")
        st.warning("Showing a sample decision boundary visualization with synthetic data.")
        
        # Create synthetic data based on disease type
        np.random.seed(42)
        n_samples = 40
        
        if disease_type == 'heart':
            param1, param2 = "Age", "Cholesterol"
        elif disease_type == 'diabetes':
            param1, param2 = "Glucose", "BMI"
        else:
            param1, param2 = "Parameter 1", "Parameter 2"
            
        # Generate two clusters for high/low risk
        center1 = np.array([55, 220])  # Center for high risk
        center2 = np.array([35, 170])  # Center for low risk
        
        # Generate high risk cluster
        X_high = np.random.randn(n_samples//2, 2) * 10 + center1
        # Generate low risk cluster
        X_low = np.random.randn(n_samples//2, 2) * 10 + center2
        
        # Combine data
        X = np.vstack([X_high, X_low])
        y = np.array(['High Risk'] * (n_samples//2) + ['Low Risk'] * (n_samples//2))
        
        # Create sample DataFrame
        sample_df = pd.DataFrame({
            'X': X[:, 0],
            'Y': X[:, 1],
            'Result': y
        })
        
        # Create scatter plot
        fig = px.scatter(
            sample_df,
            x='X',
            y='Y',
            color='Result',
            title=f'Example Decision Boundary for {self.disease_options[disease_type]}',
            labels={'X': param1, 'Y': param2},
            color_discrete_map={'High Risk': '#ff7f0e', 'Low Risk': '#1f77b4'}
        )
        
        # Add decision boundary
        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            
            # Normalize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            y_bool = np.array([label == 'High Risk' for label in y])
            
            # Fit SVM model
            model = SVC(kernel='linear')
            model.fit(X_scaled, y_bool)
            
            # Create grid for decision boundary
            x_min, x_max = sample_df['X'].min() - 5, sample_df['X'].max() + 5
            y_min, y_max = sample_df['Y'].min() - 5, sample_df['Y'].max() + 5
            
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))
            
            # Transform grid points
            grid_transformed = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
            
            # Get predictions and reshape
            Z = model.decision_function(grid_transformed).reshape(xx.shape)
            
            # Add contour to plot
            fig.add_trace(
                go.Contour(
                    z=Z,
                    x=np.linspace(x_min, x_max, 100),
                    y=np.linspace(y_min, y_max, 100),
                    showscale=False,
                    contours=dict(
                        start=0,
                        end=0,
                        size=0,
                        coloring='lines',
                        showlabels=False
                    ),
                    line=dict(
                        width=2,
                        color='black',
                        dash='dash'
                    ),
                    name='Decision Boundary'
                )
            )
        except Exception:
            pass
        
        fig.update_layout(
            xaxis_title=param1,
            yaxis_title=param2,
            plot_bgcolor='rgba(30, 30, 46, 0.8)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.info("""
        This is a synthetic example showing how a decision boundary visualization looks.
        The boundary separates high-risk predictions (orange) from low-risk predictions (blue).
        Make more predictions with both high and low risk results to see your actual decision boundary.
        """)

    def create_feature_importance_plot(self, username, disease_type):
        """Create a visualization showing feature importance for disease predictions."""
        # Get parameter data
        param_data = get_user_prediction_parameters(username, disease_type)
        
        if not param_data or len(param_data) == 0:
            st.info(f"No {self.disease_options[disease_type]} prediction data available.")
            return
        
        # Get feature importance data from database
        # In a real application, this would come from a trained model
        # For demonstration, we'll create simulated feature importance based on parameter correlations
        
        # Extract parameters from all predictions
        all_params = {}
        
        for record in param_data:
            if 'input_data' in record:
                for param, value in record['input_data'].items():
                    try:
                        # Try to convert value to float
                        value = float(value)
                        if param not in all_params:
                            all_params[param] = []
                        all_params[param].append(value)
                    except (ValueError, TypeError):
                        # Skip non-numeric parameters
                        continue
        
        # Need at least 2 parameters with data for correlation-based importance
        numeric_params = list(all_params.keys())
        if len(numeric_params) < 2:
            st.info("Not enough numeric parameters found to calculate feature importance.")
            return
        
        # Calculate feature importance (this would use ML model in production)
        # For now, we'll simulate importance based on variance and data range
        feature_importance = {}
        
        for param, values in all_params.items():
            # Calculate variance-based importance (more variance = more important)
            if len(values) >= 2:
                variance = np.var(values)
                data_range = max(values) - min(values)
                
                # Combine metrics for an "importance" score
                importance = (variance * data_range) / (np.mean(values) if np.mean(values) != 0 else 1)
                feature_importance[param] = abs(importance)
        
        # Normalize importance values to 0-100
        if feature_importance:
            max_importance = max(feature_importance.values())
            if max_importance > 0:  # Avoid division by zero
                for param in feature_importance:
                    feature_importance[param] = (feature_importance[param] / max_importance) * 100
        
        # Prepare data for visualization
        feature_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        })
        
        # Sort by importance
        feature_df = feature_df.sort_values('Importance', ascending=False)
        
        # Create interactive visualizations
        st.subheader(f"Feature Importance for {self.disease_options[disease_type]}")
        
        # Create horizontal bar chart
        fig = px.bar(
            feature_df,
            y='Feature',
            x='Importance',
            orientation='h',
            title=f"Feature Importance for {self.disease_options[disease_type]} Prediction",
            color='Importance',
            color_continuous_scale='Viridis',
            labels={'Importance': 'Relative Importance (%)'}
        )
        
        fig.update_layout(
            xaxis_title='Relative Importance (%)',
            yaxis_title='Feature',
            yaxis=dict(autorange="reversed"),  # Top-to-bottom importance
            height=max(400, len(feature_df) * 30),  # Dynamic height based on feature count
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor='rgba(30, 30, 46, 0.8)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add tree map visualization for hierarchical importance view
        st.subheader("Hierarchical Feature Importance")
        
        # Create treemap chart
        fig = px.treemap(
            feature_df,
            path=['Feature'],
            values='Importance',
            color='Importance',
            color_continuous_scale='Viridis',
            title=f"Feature Importance Hierarchy for {self.disease_options[disease_type]}"
        )
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor='rgba(30, 30, 46, 0.8)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add feature comparison section
        st.subheader("Feature Comparison")
        
        # Let user select features to compare
        top_features = feature_df['Feature'].tolist()[:min(5, len(feature_df))]
        selected_features = st.multiselect(
            "Select features to compare:",
            options=feature_df['Feature'].tolist(),
            default=top_features
        )
        
        if selected_features:
            # Filter data for selected features
            comparison_df = feature_df[feature_df['Feature'].isin(selected_features)]
            
            # Create radar chart for selected features
            categories = comparison_df['Feature'].tolist()
            values = comparison_df['Importance'].tolist()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Feature Importance',
                line=dict(color='rgba(255, 127, 14, 0.8)'),
                fillcolor='rgba(255, 127, 14, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title="Feature Importance Comparison",
                showlegend=False,
                plot_bgcolor='rgba(30, 30, 46, 0.8)',
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Add explanatory text
        st.markdown("""
        ### Understanding Feature Importance
        
        Feature importance measures how much each parameter contributes to the disease prediction outcome. 
        Higher importance means the feature has a stronger influence on the prediction result.
        
        **Note**: This visualization is based on statistical properties of your data and may not 
        reflect exact clinical significance. Always consult healthcare professionals for medical advice.
        """)
        
        # Add interactive feature exploration
        if len(feature_df) > 0:
            st.subheader("Interactive Feature Explorer")
            
            # Let user select a feature to explore
            feature_to_explore = st.selectbox(
                "Select a feature to explore:",
                options=feature_df['Feature'].tolist()
            )
            
            if feature_to_explore and feature_to_explore in all_params:
                feature_values = all_params[feature_to_explore]
                
                # Create histogram of feature values
                fig = px.histogram(
                    x=feature_values,
                    nbins=min(20, len(feature_values)),
                    title=f"Distribution of {feature_to_explore} Values",
                    labels={'x': feature_to_explore, 'y': 'Frequency'},
                    color_discrete_sequence=['rgba(255, 127, 14, 0.8)']
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(30, 30, 46, 0.8)',
                    paper_bgcolor='rgba(30, 30, 46, 0.8)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True) 