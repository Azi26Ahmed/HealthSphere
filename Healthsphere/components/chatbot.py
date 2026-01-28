import streamlit as st 
import spacy
from fuzzywuzzy import process, fuzz
from components.qa_data import predefined_qna
from streamlit_lottie import st_lottie
import json
import os
import requests
from bs4 import BeautifulSoup
import re
from components.database import save_chat_history
from components.utils import load_lottie_file

# Load your Lottie animation file
chat_animation_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Animation", "chat.json")
chat_animation = load_lottie_file(chat_animation_path)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Preprocess text by converting to lowercase and removing extra whitespace."""
    return " ".join(text.lower().split())

def find_best_match(user_input, qna_dict):
    """Find the best matching question using improved fuzzy matching and keyword analysis."""
    user_input = preprocess_text(user_input)
    
    # Check if it's a very short query like "what is X" for a non-medical term
    tokens = user_input.lower().split()
    if len(tokens) <= 3 and tokens[0] == "what" and tokens[1] == "is":
        # Check if the last word (the topic) is a common medical term
        last_word = tokens[-1] if len(tokens) > 2 else ""
        medical_terms = ['disease', 'condition', 'syndrome', 'disorder', 'virus', 'bacteria', 
                        'infection', 'heart', 'lung', 'liver', 'kidney', 'brain', 'blood', 
                        'cancer', 'tumor', 'diabetes', 'pressure', 'cholesterol', 'stroke',
                        'asthma', 'arthritis', 'alzheimer', 'parkinsons', 'hypertension',
                        'covid', 'vaccine', 'medication', 'treatment', 'symptom']
        
        # If not a medical term, don't try to match with medical QA
        if last_word and last_word not in medical_terms:
            return None
    
    # Extract important keywords from user input
    user_doc = nlp(user_input)
    user_keywords = set([token.text.lower() for token in user_doc if not token.is_stop and token.is_alpha])
    
    # First try exact keyword matching for better precision
    exact_matches = [] 
    for question in qna_dict.keys():
        question_lower = question.lower()
        # Check if any important user keyword appears in the question
        if any(keyword in question_lower for keyword in user_keywords):
            exact_matches.append(question)
    
    # If we have exact keyword matches, perform fuzzy matching only on those
    if exact_matches:
        candidate_questions = exact_matches
    else:
        candidate_questions = list(qna_dict.keys())
    
    # Use improved fuzzy matching with token_set_ratio which handles word order differences better
    matches = process.extract(user_input, candidate_questions, 
                             scorer=fuzz.token_set_ratio, 
                             limit=5)
    
    # Filter matches with score above higher threshold (85 instead of 80)
    good_matches = [(q, s) for q, s in matches if s > 85]
    
    if good_matches:
        # Return the best match
        return good_matches[0][0]
    
    # If no good matches with higher threshold, try with lower threshold for medical questions
    if any(med_term in user_input for med_term in ['disease', 'condition', 'symptom', 'treatment',
                                                 'heart', 'blood', 'cancer', 'diabetes']):
        good_matches = [(q, s) for q, s in matches if s > 70]
        
        if good_matches:
            return good_matches[0][0]
    
    # If still no good matches, use keyword-based matching as fallback
    best_score = 0
    best_question = None
    
    for question in qna_dict.keys():
        question_doc = nlp(question.lower())
        question_keywords = set([token.text.lower() for token in question_doc if not token.is_stop and token.is_alpha])
        
        # Calculate keyword overlap and weight by the importance of matching keywords
        overlap = len(user_keywords.intersection(question_keywords))
        if overlap > best_score:
            best_score = overlap
            best_question = question
    
    # Only return keyword match if we have at least two keywords overlap
    if best_score > 1:
        return best_question
    
    return None  # Return None if no good match is found

def search_web(query):
    """Search the web for information if not found in predefined QA"""
    try:
        # Clean query for search
        search_query = query.replace(' ', '+')
        
        # Try multiple search engines with fallbacks
        search_results = None
        
        # First attempt: Try Google search
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(
                f"https://www.google.com/search?q={search_query}", 
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract search results
                results = []
                
                # Try to get featured snippets or knowledge panels
                featured_snippets = soup.select('.kp-wholepage, .g .VwiC3b')
                for snippet in featured_snippets[:2]:
                    snippet_text = snippet.get_text(strip=True)
                    if len(snippet_text) > 50:  # Only include substantial snippets
                        results.append({
                            "title": "Featured Information",
                            "snippet": snippet_text
                        })
                
                # Get regular search results
                titles = soup.select('.g .LC20lb')
                descriptions = soup.select('.g .VwiC3b')
                
                # Combine titles and descriptions
                for i in range(min(len(titles), len(descriptions), 3)):
                    title = titles[i].get_text(strip=True)
                    snippet = descriptions[i].get_text(strip=True)
                    if title and snippet and len(snippet) > 20:
                        results.append({"title": title, "snippet": snippet})
                
                if results:
                    search_results = results
        except Exception as e:
            print(f"Google search failed: {str(e)}")
        
        # Second attempt: Try Bing search if Google failed
        if not search_results:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(
                    f"https://www.bing.com/search?q={search_query}", 
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract search results
                    results = []
                    search_results_elements = soup.select('.b_algo')
                    
                    for result in search_results_elements[:3]:
                        title_elem = result.select_one('h2')
                        snippet_elem = result.select_one('.b_caption p')
                        
                        if title_elem and snippet_elem:
                            title = title_elem.get_text(strip=True)
                            snippet = snippet_elem.get_text(strip=True)
                            results.append({"title": title, "snippet": snippet})
                    
                    if results:
                        search_results = results
            except Exception as e:
                print(f"Bing search failed: {str(e)}")
        
        # Third attempt: Try DuckDuckGo search if both Google and Bing failed
        if not search_results:
            try:
                response = requests.get(
                    f"https://html.duckduckgo.com/html/?q={search_query}", 
                    headers=headers,
                    timeout=5
                )
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract search results
                    results = []
                    for result in soup.select('.result__body'):
                        title_elem = result.select_one('.result__title')
                        snippet_elem = result.select_one('.result__snippet')
                        
                        if title_elem and snippet_elem:
                            title = title_elem.get_text(strip=True)
                            snippet = snippet_elem.get_text(strip=True)
                            results.append({"title": title, "snippet": snippet})
                        
                        if len(results) >= 3:  # Get top 3 results
                            break
                    
                    if results:
                        search_results = results
            except Exception as e:
                print(f"DuckDuckGo search failed: {str(e)}")
        
        # Format and return results if we got any
        if search_results:
            return format_search_results(query, search_results)
        
        # If all search attempts failed, return a generic message
        return f"""I couldn't find specific information about '{query}' due to connection issues.

I typically can answer questions about:
• Medical conditions and treatments
• General knowledge topics
• Current events and information

Please try asking in a different way or check your internet connection."""
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return f"I'm sorry, I encountered an error while searching for information about '{query}'. Please try asking in a different way."

def format_search_results(query, results):
    """Format search results into a readable answer"""
    answer = f"Here's what I found about '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        answer += f"{i}. **{result['title']}**\n{result['snippet']}\n\n"
    
    answer += "This information comes from online sources."
    return answer

def apply_chat_css():
    """Apply custom CSS to make the chat interface look more like ChatGPT"""
    st.markdown("""
    <style>
    /* Reset all default Streamlit spacing */
    .e1f1d6gn3, .css-1fv8s86, .e1nzilvr1, .e16nr0p34, .css-ocqkz7, .css-4yfn50, .css-1kyxreq,
    .css-5rimss, .css-1avcm0n, .css-18e3th9, .css-1inwz65, .css-2trqyj {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Root container */
    .block-container {
        padding: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Message container */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        height: 400px;
        overflow-y: auto;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        background-color: #f9f9f9;
        padding: 10px;
        margin-top: 5px !important;
    }
    
    /* Message styling */
    .chat-message {
        display: flex;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        width: 80%;
        line-height: 1.5;
    }
    
    /* User message (right-aligned) */
    .user-message {
        background-color: #2c533d;
        margin-left: auto;
        margin-right: 0;
        border-radius: 12px 12px 0 12px;
        border: 1px solid #1e3b2b;
        color: #ffffff;
    }
    
    /* Bot message (left-aligned) */
    .bot-message {
        background-color: #383b45;
        margin-right: auto;
        margin-left: 0;
        border-radius: 12px 12px 12px 0;
        border: 1px solid #2a2d35;
        color: #ffffff;
    }
    
    /* Remove vertical spacing in Streamlit elements */
    .stText p {
        margin-bottom: 0 !important;
    }
    
    /* Tight input styling */
    .input-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 5px 0 !important;
    }
    
    .input-row .stTextInput {
        flex-grow: 1;
    }
    
    /* Custom button and input styling */
    .stButton > button {
        background-color: #2c533d;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-size: 14px;
    }
    
    .stTextInput > div > div > input {
        border: 1px solid #ccc;
        border-radius: 20px;
        padding: 0.5rem 1rem;
    }
    
    /* Hide empty divs */
    div:empty {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

def show_chatbot():
    """Display the chatbot interface in Streamlit."""
    apply_chat_css()
    
    # Header with animation
    header_col1, header_col2 = st.columns([1, 5])
    with header_col1:
        st_lottie(chat_animation, height=100, width=100, speed=10)
    with header_col2:
        st.title("HealthSphere Chatbot")
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write("Your Comprehensive Health Assistant")
        # Add Clear Chat button
    with col2:
        if st.button("Clear Chat History"):
            # Clear chat history
            st.session_state.current_chat = []
        # Display success message
            st.success("Chat history cleared successfully!")
    
    

    # Initialize session state for chat history
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = []

    # Create a single-column layout for the input area
    user_input = st.text_input("", placeholder="Type your message here...", key="user_input")
    send_button = st.button("Send")
    
    # Render chat container with a small gap
    st.markdown('<div style="height:5px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display messages
    for message in st.session_state.current_chat:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            # Handle newlines in bot responses for better formatting
            formatted_content = message["content"].replace('\n', '<br>')
            st.markdown(f'<div class="chat-message bot-message">{formatted_content}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle input and generate response
    if send_button and user_input:
        # Special case for short "what is X" queries - direct to web search for non-medical terms
        tokens = user_input.lower().split()
        force_web_search = False
        
        if len(tokens) <= 4 and "what" in tokens and "is" in tokens:
            # Check for common non-medical objects
            common_objects = ['cake', 'bike', 'car', 'house', 'computer', 'phone', 'tree', 
                             'book', 'movie', 'music', 'game', 'sport', 'food', 'animal',
                             'country', 'city', 'language', 'ocean', 'mountain', 'river',
                             'planet', 'star', 'universe', 'science', 'history', 'art']
            
            if any(obj in tokens for obj in common_objects):
                force_web_search = True
        
        # Try to find match in predefined QA
        matched_question = None if force_web_search else find_best_match(user_input, predefined_qna)
        
        # Get the match confidence score
        if matched_question:
            score = fuzz.token_set_ratio(preprocess_text(user_input), preprocess_text(matched_question))
        else:
            score = 0
        
        # If we have a good match (above 80% confidence), use the predefined answer
        if matched_question and score > 80:
            bot_response = predefined_qna[matched_question]
            # Find related questions
            related_questions = []
            
            # Only show related questions that are in the same category
            # Extract the first few words which typically contain the category
            category_words = matched_question.lower().split()[:3]
            
            for q in predefined_qna.keys():
                # Skip the matched question itself
                if q == matched_question:
                    continue
                    
                # Check if it might be in the same category
                if any(word in q.lower() for word in category_words):
                    related_questions.append((q, 0))  # Score doesn't matter here
                    if len(related_questions) >= 3:
                        break
            
            # If we didn't find related by category, use fuzzy matching
            if not related_questions:
                related_questions = process.extract(matched_question, 
                                                   [q for q in predefined_qna.keys() if q != matched_question], 
                                                   limit=3)
            
            # Add related questions to response
            if related_questions:
                bot_response += "\n\nRelated questions you might be interested in:\n"
                for q, score in related_questions:
                    bot_response += f"- {q}\n"
        else:
            # Web search for information (for both medical and non-medical queries)
            bot_response = search_web(user_input)

        # Store conversation in session state
        st.session_state.current_chat.append({"role": "user", "content": user_input})
        st.session_state.current_chat.append({"role": "assistant", "content": bot_response})
        
        # Save chat to database if user is logged in
        if "username" in st.session_state:
            save_chat_history(st.session_state.username, st.session_state.current_chat)
        
        # Refresh the page to show the new messages in the chat container
        st.rerun()