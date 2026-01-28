import streamlit as st
from components.database import get_prediction_stats
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from components.utils import safe_parse_datetime

def show_stats():
    try:
        st.title("Health Statistics Dashboard")
        st.write("View and analyze your health prediction history")
        
        username = st.session_state.username
        stats = get_prediction_stats(username)
        
        if stats:
            # Header section with key metrics in cards
            st.markdown("## Key Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div style="background-color:#e8f4f8;padding:20px;border-radius:10px;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.1);border:1px solid #d0e8f2;">
                    <h3 style="color:#333333;margin-bottom:10px;font-weight:600;">Total Predictions</h3>
                    <h1 style="color:#1f77b4;font-size:2.8em;margin:5px 0;">{}</h1>
                </div>
                """.format(stats['total_predictions']), unsafe_allow_html=True)
            
                # Calculate high risk count from risk_distribution
                high_risk_count = 0
                for disease, risks in stats.get('risk_distribution', {}).items():
                    high_risk_count += risks.get('high_risk', 0)
            
            with col2:
                high_risk_percentage = (high_risk_count / stats['total_predictions']) * 100 if stats['total_predictions'] > 0 else 0
                st.markdown("""
                <div style="background-color:#fef6e8;padding:20px;border-radius:10px;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.1);border:1px solid #fae0bc;">
                    <h3 style="color:#333333;margin-bottom:10px;font-weight:600;">High Risk Predictions</h3>
                    <h1 style="color:#ff7f0e;font-size:2.8em;margin:5px 0;">{} <span style="font-size:16px;color:#666;">({:.1f}%)</span></h1>
                </div>
                """.format(high_risk_count, high_risk_percentage), unsafe_allow_html=True)
            
            with col3:
                # Calculate average confidence
                avg_confidence = stats.get('avg_confidence', 0)
                # Ensure avg_confidence is a number, not None
                if avg_confidence is None:
                    avg_confidence = 0
                st.markdown("""
                <div style="background-color:#e9f5e9;padding:20px;border-radius:10px;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.1);border:1px solid #c8e6c9;">
                    <h3 style="color:#333333;margin-bottom:10px;font-weight:600;">Avg Confidence</h3>
                    <h1 style="color:#2ca02c;font-size:2.8em;margin:5px 0;">{:.1f}%</h1>
                </div>
                """.format(avg_confidence), unsafe_allow_html=True)
            
            # Pie chart for disease distribution
            st.markdown("## Disease Distribution")
            disease_data = []
            for disease, count in stats['disease_distribution'].items():
                disease_name = disease.replace('_', ' ').title()
                disease_data.append({"Disease": disease_name, "Count": count})
            
            if disease_data:
                disease_df = pd.DataFrame(disease_data)
                fig = px.pie(disease_df, values='Count', names='Disease', 
                            title='Distribution of Disease Types',
                            color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(margin=dict(t=40, b=40, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk distribution by disease
            st.markdown("## Risk Assessment by Disease Type")
            
            risk_data = []
            for disease, risks in stats.get('risk_distribution', {}).items():
                disease_name = disease.replace('_', ' ').title()
                high_risk = risks.get('high_risk', 0)
                low_risk = risks.get('low_risk', 0)
                risk_data.append({"Disease": disease_name, "High Risk": high_risk, "Low Risk": low_risk})
            
            if risk_data:
                risk_df = pd.DataFrame(risk_data)
                
                # Convert to long format for stacked bar chart
                risk_df_long = pd.melt(risk_df, 
                                    id_vars=['Disease'],
                                    value_vars=['High Risk', 'Low Risk'],
                                    var_name='Risk Level', 
                                    value_name='Count')
                
                fig = px.bar(risk_df_long, 
                            x='Disease', 
                            y='Count', 
                            color='Risk Level',
                            title='Risk Distribution by Disease Type',
                            barmode='stack',
                            color_discrete_map={'High Risk': '#ff7f0e', 'Low Risk': '#1f77b4'})
                
                fig.update_layout(xaxis_title='Disease Type',
                                yaxis_title='Number of Predictions',
                                legend_title='Risk Level')
                                    
                st.plotly_chart(fig, use_container_width=True)
            
            # Monthly trends visualization
            st.markdown("## Prediction Trends Over Time")
            monthly_data = []
            
            for month, data in stats.get('monthly_trends', {}).items():
                total = data.get('total', 0)
                high_risk = data.get('high_risk', 0)
                low_risk = total - high_risk
                try:
                    # Parse month as "MM/YYYY"
                    month_date = datetime.strptime(month, "%m/%Y")
                except ValueError:
                    # Fallback for older format "YYYY-MM"
                    try:
                        month_date = datetime.strptime(month, "%Y-%m")
                    except ValueError:
                        # If all parsing fails, use current date
                        month_date = datetime.now()
                
                monthly_data.append({
                    "Month": month_date,
                    "Total": total,
                    "High Risk": high_risk,
                    "Low Risk": low_risk
                })
            
            if monthly_data:
                monthly_df = pd.DataFrame(monthly_data).sort_values('Month')
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=monthly_df['Month'], 
                    y=monthly_df['Total'],
                    mode='lines+markers',
                    name='Total Predictions'
                ))
                fig.add_trace(go.Scatter(
                    x=monthly_df['Month'], 
                    y=monthly_df['High Risk'],
                    mode='lines+markers',
                    name='High Risk',
                    line=dict(color='#ff7f0e')
                ))
                fig.add_trace(go.Scatter(
                    x=monthly_df['Month'], 
                    y=monthly_df['Low Risk'],
                    mode='lines+markers',
                    name='Low Risk',
                    line=dict(color='#1f77b4')
                ))
                
                fig.update_layout(
                    title='Monthly Prediction Trends',
                    xaxis_title='Month',
                    yaxis_title='Number of Predictions',
                    legend_title='Prediction Type',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent Activity section
            st.markdown("## Recent Activity")
            # Get all predictions sorted by timestamp (newest first)
            from components.database import get_user_predictions as get_predictions, delete_prediction
            all_predictions_data = get_predictions(username)
            
            if all_predictions_data and "predictions" in all_predictions_data:
                # Access the predictions list from the returned dictionary
                all_predictions = all_predictions_data["predictions"]
                
                # No need to sort as they are already sorted by timestamp in the database query
                
                for pred in all_predictions:
                    pred_disease_type = pred['disease_type']
                    disease_name = pred_disease_type.replace('_', ' ').title()
                    timestamp = pred['timestamp']
                    
                    # Ensure timestamp is datetime object
                    if isinstance(timestamp, str):
                        try:
                            timestamp = safe_parse_datetime(timestamp)
                        except:
                            pass
                    
                    # Create a container for each prediction
                    with st.container():
                        # Create two columns: one for content, one for delete button
                        col1, col2 = st.columns([9, 1])
                        
                        with col1:
                            with st.expander(
                                f"{disease_name} - {timestamp.strftime('%d/%m/%Y %H:%M')}"
                            ):
                                # Display prediction result with appropriate color
                                if pred['prediction_result']:
                                    st.error("Result: High Risk")
                                else:
                                    st.success("Result: Low Risk")
                                    
                                # Display confidence with better formatting
                                probability_value = pred.get('probability', pred.get('prediction_probability', 0))
                                st.write(f"Confidence: {probability_value*100:.1f}%" if probability_value is not None else "Confidence: N/A")
                                
                                # Display input parameters if available
                                if 'input_data' in pred:
                                    st.subheader("Input Parameters")
                                    input_data = pred['input_data']
                                    
                                    # Create two columns for better layout
                                    col1, col2 = st.columns(2)
                                    use_col1 = True
                                    
                                    for param, value in input_data.items():
                                        current_col = col1 if use_col1 else col2
                                        use_col1 = not use_col1
                                        
                                        with current_col:
                                            st.write(f"**{param}**: {value}")
                                    
                                    # Add a divider for better visual separation
                                    st.markdown("---")
                        
                        with col2:
                            # Add delete button
                            if st.button("🗑️", key=f"delete_{pred['timestamp']}"):
                                success, message = delete_prediction(
                                    username=st.session_state.username,
                                    disease_type=pred_disease_type,
                                    timestamp=timestamp
                                )
                                
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
            else:
                st.info("No predictions found")
        else:
            st.info("No prediction statistics available yet.")
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")
        st.exception(e)  # Debugging: show detailed error