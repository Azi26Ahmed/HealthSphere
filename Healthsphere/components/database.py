from pymongo import MongoClient
from dotenv import load_dotenv
import os
import bcrypt
from datetime import datetime
import gridfs
from components.utils import safe_parse_datetime

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['healthsphere_db']

# Initialize GridFS for storing images
fs = gridfs.GridFS(db)

def init_db():
    """Initialize database collections if they don't exist"""
    collections = [
        'users',
        'predictions',
        'diabetes_predictions',
        'heart_predictions',
        'stroke_predictions',
        'kidney_predictions',
        'liver_predictions',
        'parkinsons_predictions',
        'brain_tumor_predictions',
        'user_files',  
        'chat_history'
    ]
    
    for collection in collections:
        if collection not in db.list_collection_names():
            db.create_collection(collection)

def register_user(username, password):
    """Register a new user"""
    users = db.users
    
    # Check if username already exists
    if users.find_one({'username': username}):
        return False, "Username already exists"
    
    # Hash the password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    # Create user document
    user = {
        'username': username,
        'password': hashed_password,
        'created_at': datetime.now(),
        'last_login': datetime.now()
    }
    
    users.insert_one(user)
    return True, "User registered successfully"

def verify_user(username, password):
    """Verify user credentials"""
    users = db.users
    user = users.find_one({'username': username})
    
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        # Update last login time
        users.update_one(
            {'username': username},
            {'$set': {'last_login': datetime.now()}}
        )
        return True, user
    return False, None

def save_prediction(username, disease_type, input_data, prediction_result, file_data=None):
    """Save prediction results for a user"""
    # Get the appropriate collection for the disease type
    collection_name = f"{disease_type.lower().replace(' ', '_')}_predictions"
    predictions = db[collection_name]
    
    # Create the prediction document
    prediction_doc = {
        'username': username,
        'timestamp': datetime.now(),
        'input_data': input_data,
        'prediction_result': prediction_result,
        'confidence': prediction_result.get('confidence', 0)
    }
    
    # Handle file data (e.g., for brain tumor MRI scans)
    if file_data:
        file_id = fs.put(file_data, filename=f"{username}_{datetime.now().timestamp()}")
        prediction_doc['file_id'] = file_id
    
    # Store in disease-specific collection
    predictions.insert_one(prediction_doc)
    
    # Store in general predictions collection
    general_prediction = {
        'username': username,
        'disease_type': disease_type,
        'timestamp': datetime.now(),
        'prediction_result': prediction_result,
        'confidence': prediction_result.get('confidence', 0)
    }
    db.predictions.insert_one(general_prediction)

def get_user_predictions(username, disease_type=None, start_date=None, end_date=None, page=1, per_page=20):
    """Get predictions for a specific user with optional filters and pagination
    
    Args:
        username: Username to filter predictions
        disease_type: Optional disease type filter
        start_date: Optional start date filter
        end_date: Optional end date filter
        page: Page number (starting from 1)
        per_page: Number of items per page
        
    Returns:
        Dictionary containing predictions and pagination info
    """
    query = {'username': username}
    
    if start_date and end_date:
        query['timestamp'] = {
            '$gte': start_date,
            '$lte': end_date
        }
    
    # Calculate skip value for pagination
    skip = (page - 1) * per_page
    
    try:
        if disease_type:
            collection_name = f"{disease_type.lower().replace(' ', '_')}_predictions"
            predictions = db[collection_name]
            
            # Get total count for pagination
            total_count = predictions.count_documents(query)
            
            # Get paginated results with sort
            results = list(predictions.find(query).sort("timestamp", -1).skip(skip).limit(per_page))
            
            return {
                "predictions": results,
                "pagination": {
                    "total": total_count,
                    "page": page,
                    "per_page": per_page,
                    "pages": (total_count + per_page - 1) // per_page  # Ceiling division
                }
            }
        else:
            predictions = db.predictions
            
            # Get total count for pagination
            total_count = predictions.count_documents(query)
            
            # Get paginated results with sort
            results = list(predictions.find(query).sort("timestamp", -1).skip(skip).limit(per_page))
            
            return {
                "predictions": results,
                "pagination": {
                    "total": total_count,
                    "page": page,
                    "per_page": per_page,
                    "pages": (total_count + per_page - 1) // per_page  # Ceiling division
                }
            }
    except Exception as e:
        print(f"Error getting predictions: {str(e)}")
        return {
            "predictions": [],
            "pagination": {
                "total": 0,
                "page": page,
                "per_page": per_page,
                "pages": 0
            },
            "error": str(e)
        }

def get_prediction_stats(username):
    """
    Get statistics about a user's prediction history
    """
    try:
        # Get predictions directly from the predictions collection instead of user document
        predictions_data = list(db.predictions.find({"username": username}))
        if not predictions_data:
            return None

        # Initialize statistics
        stats = {
            "total_predictions": len(predictions_data),
            "disease_distribution": {},
            "risk_distribution": {},
            "monthly_trends": {},
            "avg_confidence": 0
        }

        # Process each prediction
        total_confidence = 0
        valid_confidence_count = 0

        for pred in predictions_data:
            disease_type = pred.get("disease_type", "unknown")
            timestamp = pred.get("timestamp")
            
            # Ensure timestamp is a datetime object
            if isinstance(timestamp, str):
                try:
                    timestamp = safe_parse_datetime(timestamp)
                except:
                    # If parsing fails, skip this prediction for trends
                    continue
            
            # Calculate month key for trends - change to dd/mm/yyyy format for month display
            month_key = timestamp.strftime("%m/%Y")
            
            # Count by disease type
            if disease_type not in stats["disease_distribution"]:
                stats["disease_distribution"][disease_type] = 0
            stats["disease_distribution"][disease_type] += 1
            
            # Initialize risk distribution for this disease if not exists
            if disease_type not in stats["risk_distribution"]:
                stats["risk_distribution"][disease_type] = {"high_risk": 0, "low_risk": 0}
            
            # Count by risk level
            risk_type = "high_risk" if pred.get("prediction_result", False) else "low_risk"
            stats["risk_distribution"][disease_type][risk_type] += 1
            
            # Initialize monthly trend for this month if not exists
            if month_key not in stats["monthly_trends"]:
                stats["monthly_trends"][month_key] = {"total": 0, "high_risk": 0}
            
            # Update monthly trends
            stats["monthly_trends"][month_key]["total"] += 1
            if pred.get("prediction_result", False):
                stats["monthly_trends"][month_key]["high_risk"] += 1
            
            # Sum confidence for average calculation
            probability = pred.get("probability", pred.get("confidence", 0))
            if probability is not None:
                total_confidence += probability * 100  # Convert to percentage
                valid_confidence_count += 1
        
        # Calculate average confidence
        if valid_confidence_count > 0:
            stats["avg_confidence"] = total_confidence / valid_confidence_count
        
        return stats
        
    except Exception as e:
        print(f"Error in get_prediction_stats: {str(e)}")
        return None

def get_user_prediction_parameters(username, disease_type):
    """Get all input parameters used for predictions of a specific disease type"""
    collection_name = f"{disease_type.lower().replace(' ', '_')}_predictions"
    predictions = db[collection_name]
    return list(predictions.find(
        {'username': username},
        {'input_data': 1, 'timestamp': 1, '_id': 0}
    ))

def get_prediction_history_by_date(username, start_date, end_date, disease_type=None):
    """Get prediction history within a date range"""
    return get_user_predictions(username, disease_type, start_date, end_date)

def get_disease_specific_stats(username, disease_type):
    """Get detailed statistics for a specific disease type"""
    collection_name = f"{disease_type.lower().replace(' ', '_')}_predictions"
    predictions = db[collection_name]
    
    # Get all predictions for analysis
    user_predictions = list(predictions.find({'username': username}))
    
    if not user_predictions:
        return None
    
    # Calculate statistics
    total_predictions = len(user_predictions)
    positive_predictions = sum(1 for p in user_predictions if p['prediction_result'])
    avg_confidence = sum(float(p.get('confidence', 0)) for p in user_predictions) / total_predictions
    
    # Calculate parameter trends if available
    parameter_trends = {}
    if user_predictions[0].get('input_data'):
        for param in user_predictions[0]['input_data'].keys():
            if param != 'image':  # Skip image data
                values = [float(p['input_data'].get(param, 0)) for p in user_predictions if param in p['input_data']]
                if values:
                    parameter_trends[param] = {
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'trend': 'increasing' if values[-1] > values[0] else 'decreasing'
                    }
    
    return {
        'total_predictions': total_predictions,
        'positive_predictions': positive_predictions,
        'average_confidence': avg_confidence,
        'parameter_trends': parameter_trends
    }

def get_parameter_trends(username, disease_type, parameter_name):
    """Get trends for a specific parameter over time"""
    collection_name = f"{disease_type.lower().replace(' ', '_')}_predictions"
    predictions = db[collection_name]
    
    # Get all predictions with the specified parameter
    pipeline = [
        {'$match': {'username': username}},
        {'$project': {
            'timestamp': 1,
            'parameter_value': f'$input_data.{parameter_name}',
            'prediction_result': 1,
            'confidence': 1
        }},
        {'$sort': {'timestamp': 1}}
    ]
    
    results = list(predictions.aggregate(pipeline))
    
    if not results:
        return None
    
    # Convert parameter values to float where possible
    for result in results:
        try:
            result['parameter_value'] = float(result['parameter_value'])
        except (ValueError, TypeError):
            # Keep as is if not numeric
            pass
    
    return results

def update_username(old_username, new_username):
    """Update a user's username with optimized database operations"""
    if db.users.find_one({'username': new_username}):
        return False, "Username already exists"
    
    # Start a session for better transaction management
    session = client.start_session()
    
    try:
        with session.start_transaction():
            # Update username in users collection
            result = db.users.update_one(
                {'username': old_username},
                {'$set': {'username': new_username}}
            )
            
            if result.modified_count == 0:
                # No user found with old_username
                return False, "User not found"
                
            # Get all collection names that might contain user data
            collections = db.list_collection_names()
            prediction_collections = [c for c in collections if c.endswith('_predictions') or c == 'predictions']
            
            # Only update collections where the user has data
            for collection_name in prediction_collections:
                # Check if user has data in this collection
                if db[collection_name].count_documents({'username': old_username}) > 0:
                    # Update username in collection
                    db[collection_name].update_many(
                        {'username': old_username},
                        {'$set': {'username': new_username}}
                    )
            
            # Update chat history if it exists
            if 'chat_history' in collections and db.chat_history.count_documents({'username': old_username}) > 0:
                db.chat_history.update_many(
                    {'username': old_username},
                    {'$set': {'username': new_username}}
                )
                
            return True, "Username updated successfully"
    except Exception as e:
        return False, f"Error updating username: {str(e)}"
    finally:
        session.end_session()
        
def update_password(username, new_password):
    """Update a user's password"""
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), salt)
    
    result = db.users.update_one(
        {'username': username},
        {'$set': {'password': hashed_password}}
    )
    
    if result.modified_count > 0:
        return True, "Password updated successfully"
    return False, "Password update failed"

def get_user_chat_history(username):
    """Get chat history for a user"""
    try:
        chat_history = list(db.chat_history.find({"username": username}).sort("timestamp", -1))
        return chat_history
    except:
        # Create the collection if it doesn't exist
        if 'chat_history' not in db.list_collection_names():
            db.create_collection('chat_history')
        return []

def save_chat_history(username, messages):
    """Save a new chat session to history"""
    try:
        # Create a new chat history entry
        chat_entry = {
            "username": username,
            "timestamp": datetime.now(),
            "messages": messages
        }
        
        # Insert the entry into the chat_history collection
        db.chat_history.insert_one(chat_entry)
        return True
    except Exception as e:
        print(f"Error saving chat history: {str(e)}")
        return False

def delete_chat_history(username):
    """Delete all chat history for a user"""
    try:
        result = db.chat_history.delete_many({"username": username})
        return True, f"Deleted {result.deleted_count} chat history entries"
    except Exception as e:
        return False, f"Error deleting chat history: {str(e)}"

def delete_prediction(username, disease_type, timestamp):
    try:
        # Convert timestamp to string for MongoDB query if it's a datetime object
        if not isinstance(timestamp, str):
            timestamp = timestamp.isoformat()

        # Find the user document
        result = db.users.update_one(
            {"username": username},
            {"$pull": {"predictions": {"disease_type": disease_type, "timestamp": timestamp}}}
        )

        if result.modified_count > 0:
            return True, "Prediction deleted successfully"
        else:
            return False, "Failed to delete prediction"
    except Exception as e:
        print(f"Error in delete_prediction: {str(e)}")
        return False, f"Error: {str(e)}"