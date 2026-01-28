import streamlit as st
import bcrypt
from datetime import datetime, timedelta
from components.database import db, get_user_predictions, get_user_prediction_parameters, get_user_chat_history, delete_chat_history
from components.utils import safe_parse_datetime

def show_settings():
    """Show the settings page with profile settings, prediction history, and chat history"""
    username = st.session_state.get("username", "Guest")
    
    try:
        settings_tabs = st.tabs(["Profile Settings", "Prediction History", "Chat History"])
        
        with settings_tabs[0]:
            st.subheader("Profile Settings")
            new_username = st.text_input("Change your username:", value=username)
            if st.button("Update Username"):
                try:
                    # Username update logic
                    current_user = db.users.find_one({"username": username})
                    if current_user:
                        # Check if new username already exists
                        if new_username != username and db.users.find_one({"username": new_username}):
                            st.error("Username already exists. Please choose a different one.")
                        else:
                            # Update username in a more efficient way
                            try:
                                # Limit collections to update to reduce load
                                collections = [
                                    'users', 'predictions'
                                ]
                                
                                for collection in collections:
                                    try:
                                        db[collection].update_many(
                                            {"username": username},
                                            {"$set": {"username": new_username}}
                                        )
                                    except Exception as coll_err:
                                        st.warning(f"Minor issue updating {collection}: {str(coll_err)}")
                                
                                # Update specific disease collections only if they have data
                                disease_collections = [
                                    'diabetes_predictions', 'heart_predictions', 'stroke_predictions', 
                                    'kidney_predictions', 'liver_predictions', 'parkinsons_predictions'
                                ]
                                
                                for collection in disease_collections:
                                    if db[collection].count_documents({"username": username}) > 0:
                                        db[collection].update_many(
                                            {"username": username},
                                            {"$set": {"username": new_username}}
                                        )
                                
                                # Update session state
                                st.session_state.username = new_username
                                st.success(f"Username updated to {new_username}!")
                            except Exception as update_err:
                                st.error(f"Error updating username: {str(update_err)}")
                except Exception as e:
                    st.error(f"Database error: {str(e)}")

            st.subheader("Change Password")
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.button("Update Password"):
                try:
                    # Password update logic
                    if not current_password or not new_password or not confirm_password:
                        st.error("Please fill in all password fields.")
                    elif new_password != confirm_password:
                        st.error("New passwords do not match.")
                    else:
                        # Verify current password
                        user = db.users.find_one({"username": username})
                        if user and bcrypt.checkpw(current_password.encode('utf-8'), user['password']):
                            # Hash new password
                            salt = bcrypt.gensalt()
                            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), salt)
                            
                            # Update password
                            db.users.update_one(
                                {"username": username},
                                {"$set": {"password": hashed_password}}
                            )
                            st.success("Password updated successfully!")
                        else:
                            st.error("Current password is incorrect.")
                except Exception as e:
                    st.error(f"Error updating password: {str(e)}")

        with settings_tabs[1]:
            st.subheader("Prediction History")
            
            # Show prediction history with pagination to reduce load
            try:
                # Date range selector
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("From", datetime.now() - timedelta(days=30), format="DD/MM/YYYY")
                with col2:
                    end_date = st.date_input("To", datetime.now(), format="DD/MM/YYYY")
                
                # Disease filter
                disease_types = ["All", "Heart Disease", "Diabetes", "Stroke", 
                                "Liver Disease", "Kidney Disease", "Parkinson's Disease"]
                selected_disease = st.selectbox("Filter by Disease", disease_types)
                
                # Convert date inputs to datetime objects
                start_datetime = datetime.combine(start_date, datetime.min.time())
                end_datetime = datetime.combine(end_date, datetime.max.time())
                
                # Get predictions with limit
                disease_type_map = {
                    "Heart Disease": 'heart',
                    "Diabetes": 'diabetes',
                    "Stroke": 'stroke',
                    "Liver Disease": 'liver',
                    "Kidney Disease": 'kidney',
                    "Parkinson's Disease": 'parkinsons'
                }
                
                # Add pagination to limit the data loaded
                items_per_page = 5
                page = st.number_input("Page", min_value=1, value=1, step=1)
                skip = (page - 1) * items_per_page
                
                # Get predictions with pagination
                if selected_disease == "All":
                    # Query the database more efficiently with pagination
                    predictions = list(db.predictions.find(
                        {"username": username, "timestamp": {"$gte": start_datetime, "$lte": end_datetime}},
                        {"_id": 0}
                    ).sort("timestamp", -1).skip(skip).limit(items_per_page))
                    
                    # Get total count for pagination info
                    total_count = db.predictions.count_documents(
                        {"username": username, "timestamp": {"$gte": start_datetime, "$lte": end_datetime}}
                    )
                else:
                    disease_type = disease_type_map[selected_disease]
                    collection_name = f"{disease_type}_predictions"
                    
                    # Query the disease-specific collection with pagination
                    predictions = list(db[collection_name].find(
                        {"username": username, "timestamp": {"$gte": start_datetime, "$lte": end_datetime}},
                        {"_id": 0}
                    ).sort("timestamp", -1).skip(skip).limit(items_per_page))
                    
                    # Get total count for pagination info
                    total_count = db[collection_name].count_documents(
                        {"username": username, "timestamp": {"$gte": start_datetime, "$lte": end_datetime}}
                    )
                
                # Show pagination info
                total_pages = max(1, (total_count + items_per_page - 1) // items_per_page)
                st.write(f"Showing page {page} of {total_pages} (total records: {total_count})")
                
                # Navigation buttons
                col1, col2 = st.columns(2)
                with col1:
                    if page > 1:
                        if st.button("Previous Page"):
                            st.session_state.page = page - 1
                with col2:
                    if page < total_pages:
                        if st.button("Next Page"):
                            st.session_state.page = page + 1
                
                if predictions:
                    st.write(f"Showing {len(predictions)} predictions")
                    
                    for i, pred in enumerate(predictions):
                        try:
                            disease = pred.get('disease_type', '')
                            if selected_disease != "All":
                                disease = selected_disease
                            
                            # Convert MongoDB datetime to Python datetime if necessary
                            timestamp = pred.get('timestamp', datetime.now())
                            if isinstance(timestamp, str):
                                try:
                                    timestamp = safe_parse_datetime(timestamp)
                                except:
                                    pass
                            
                            result_text = "High Risk" if pred.get('prediction_result', False) else "Low Risk"
                            probability_value = pred.get('probability', pred.get('prediction_probability', 0))
                            
                            with st.expander(
                                f"{disease} - {timestamp.strftime('%d/%m/%Y %H:%M')}"
                            ):
                                st.write("Result:", result_text)
                                st.write(f"Confidence: {probability_value*100:.1f}%")
                                
                                # If disease-specific details are available, show them
                                if 'input_data' in pred:
                                    st.subheader("Input Parameters")
                                    for param, value in pred['input_data'].items():
                                        st.write(f"{param}: {value}")
                        except Exception as pred_err:
                            st.warning(f"Error displaying prediction: {str(pred_err)}")
                else:
                    st.info("No predictions found for the selected criteria.")
            except Exception as e:
                st.error(f"Error loading prediction history: {str(e)}")

        with settings_tabs[2]:
            st.subheader("Chat History")
            
            # Add button to delete all chat history
            try:
                if st.button("Delete All Chat History"):
                    success, message = delete_chat_history(username)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                
                # Show chat history with pagination
                items_per_page = 5
                chat_page = st.number_input("Chat Page", min_value=1, value=1, step=1, key="chat_page")
                chat_skip = (chat_page - 1) * items_per_page
                
                # Get chat history with pagination
                try:
                    # Get total count
                    total_chats = db.chat_history.count_documents({"username": username})
                    
                    # Get paginated chat history
                    chats = list(db.chat_history.find(
                        {"username": username},
                        {"_id": 0}
                    ).sort("timestamp", -1).skip(chat_skip).limit(items_per_page))
                    
                    # Show pagination info
                    total_chat_pages = max(1, (total_chats + items_per_page - 1) // items_per_page)
                    st.write(f"Showing page {chat_page} of {total_chat_pages} (total chats: {total_chats})")
                    
                    # Navigation buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if chat_page > 1:
                            if st.button("Previous Chat Page"):
                                st.session_state.chat_page = chat_page - 1
                    with col2:
                        if chat_page < total_chat_pages:
                            if st.button("Next Chat Page"):
                                st.session_state.chat_page = chat_page + 1
                    
                    if chats:
                        for chat in chats:
                            try:
                                chat_time = chat.get('timestamp', datetime.now())
                                with st.expander(f"Chat - {chat_time.strftime('%d/%m/%Y %H:%M')}"):
                                    for message in chat.get('messages', []):
                                        if message.get('role') == 'user':
                                            st.markdown(f"**You:** {message.get('content', '')}")
                                        else:
                                            st.markdown(f"**AI:** {message.get('content', '')}")
                            except Exception as chat_err:
                                st.warning(f"Error displaying chat: {str(chat_err)}")
                    else:
                        st.info("No chat history found.")
                except Exception as chat_e:
                    st.error(f"Could not load chat history: {str(chat_e)}")
            except Exception as e:
                st.error(f"Error with chat history feature: {str(e)}")
    except Exception as settings_error:
        st.error(f"Error loading settings: {str(settings_error)}")
        st.info("Please refresh the page and try again.") 