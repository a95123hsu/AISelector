import streamlit as st
import pandas as pd
import openai
from supabase import create_client
import time
from postgrest.exceptions import APIError
import re

# --- Initialize Supabase ---
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        supabase = create_client(url, key)
        return supabase
    except Exception as e:
        st.error(f"Error connecting to Supabase: {str(e)}")
        return None

supabase = init_supabase()

# --- Load pump data from Supabase ---
@st.cache_data
def load_data(table_name):
    if not supabase:
        return pd.DataFrame()
    
    try:
        response = supabase.table(table_name).select("*").execute()
        df = pd.DataFrame(response.data)
        
        if not df.empty:
            # Clean and prepare the data
            # Fix column name inconsistency (CSV has "Max Head(M)" but we want "Max Head (M)")
            if "Max Head(M)" in df.columns and "Max Head (M)" not in df.columns:
                df["Max Head (M)"] = df["Max Head(M)"]
            
            # Ensure we have the required columns
            required_columns = ["Model No.", "Max Flow (LPM)"]
            head_column = "Max Head(M)" if "Max Head(M)" in df.columns else "Max Head (M)"
            
            if head_column in df.columns:
                required_columns.append(head_column)
                # Standardize to the expected name
                if head_column == "Max Head(M)":
                    df["Max Head (M)"] = df["Max Head(M)"]
            
            # Check if we have all required columns
            missing_columns = [col for col in ["Model No.", "Max Flow (LPM)", "Max Head (M)"] if col not in df.columns]
            if missing_columns:
                st.error(f"Table {table_name} is missing required columns: {missing_columns}")
                return pd.DataFrame()
            
            # Ensure numeric columns are properly typed
            numeric_columns = ["Max Flow (LPM)", "Max Head (M)"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Remove rows with NaN values in critical columns
            df = df.dropna(subset=numeric_columns)
            
            return df
            
    except APIError as e:
        st.error(f"Error accessing table {table_name}: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error with table {table_name}: {str(e)}")
        return pd.DataFrame()

# --- Helper Functions for AI Chat ---
def extract_flow_head_from_text(text):
    """Extract flow and head requirements from user text"""
    flow_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:lpm|l/min|liters?\s*per\s*minute)',
        r'flow[:\s]*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*flow'
    ]
    
    head_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?\s*(?:head|height)',
        r'head[:\s]*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?'
    ]
    
    flow = None
    head = None
    
    text_lower = text.lower()
    
    for pattern in flow_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            flow = float(match.group(1))
            break
    
    for pattern in head_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            head = float(match.group(1))
            break
    
    return flow, head

def search_pumps(df, flow=None, head=None):
    """Search for suitable pumps based on requirements"""
    if df.empty:
        return pd.DataFrame()
    
    filtered = df.copy()
    
    if flow is not None:
        filtered = filtered[filtered["Max Flow (LPM)"] >= flow]
    
    if head is not None:
        filtered = filtered[filtered["Max Head (M)"] >= head]
    
    if not filtered.empty and flow is not None and head is not None:
        # Sort by efficiency
        filtered["flow_efficiency"] = filtered["Max Flow (LPM)"] / flow
        filtered["head_efficiency"] = filtered["Max Head (M)"] / head
        filtered["total_efficiency"] = filtered["flow_efficiency"] + filtered["head_efficiency"]
        filtered = filtered.sort_values("total_efficiency")
    
    return filtered

def generate_ai_response(user_message, df, flow=None, head=None):
    """Generate AI response based on user message and pump data"""
    if "OPENAI_API_KEY" not in st.secrets:
        return "I'd love to help you with pump selection, but I need an OpenAI API key to provide intelligent responses. For now, I can help you search for pumps if you tell me your flow (LPM) and head (meters) requirements!"
    
    try:
        # Search for relevant pumps
        filtered_pumps = search_pumps(df, flow, head)
        
        # Prepare context for AI
        context = f"""
You are a pump selection expert assistant. Help the user with their pump selection needs.

User message: "{user_message}"

Available pump database summary:
- Total pumps: {len(df)}
- Flow range: {df['Max Flow (LPM)'].min():.0f} - {df['Max Flow (LPM)'].max():.0f} LPM
- Head range: {df['Max Head (M)'].min():.1f} - {df['Max Head (M)'].max():.1f} meters

"""

        if flow is not None or head is not None:
            context += f"""
Extracted requirements:
- Flow: {flow if flow else 'Not specified'} LPM
- Head: {head if head else 'Not specified'} meters

"""

        if not filtered_pumps.empty:
            top_3 = filtered_pumps.head(3)
            context += f"""
Found {len(filtered_pumps)} matching pumps. Top 3 recommendations:
{top_3[['Model No.', 'Max Flow (LPM)', 'Max Head (M)']].to_string(index=False)}
"""
        else:
            if flow is not None and head is not None:
                context += f"No pumps found that can handle {flow} LPM and {head} meters. Suggest alternatives or ask for different requirements."

        context += """

Instructions:
1. Be helpful and conversational
2. If requirements were extracted, provide specific pump recommendations
3. If no requirements found, ask clarifying questions about flow and head needs
4. Keep responses concise but informative
5. Always be encouraging and professional
"""

        # Call OpenAI API
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": context}],
            max_tokens=400,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}. But I'm still here to help! Could you tell me your flow (LPM) and head (meters) requirements?"

# --- App UI ---
st.title("🔍 Pump Selector Assistant")
st.caption("Ask me about pump requirements and I'll help you find the perfect match! 🤖")

# Table selection in sidebar
with st.sidebar:
    st.header("📊 Database Settings")
    selected_table = st.selectbox(
        "Select Pump Data Table",
        ["pump_curve_data", "pump_selection_data"],
        help="Choose which table to use for pump data"
    )
    
    # Load and display data info
    df = load_data(selected_table)
    
    if not df.empty:
        st.success(f"✅ {len(df)} pumps loaded")
        st.metric("Flow Range", f"{df['Max Flow (LPM)'].min():.0f} - {df['Max Flow (LPM)'].max():.0f} LPM")
        st.metric("Head Range", f"{df['Max Head (M)'].min():.1f} - {df['Max Head (M)'].max():.1f} m")
    else:
        st.error("❌ No data loaded")
    
    st.header("💡 Example Questions")
    st.write("• I need a pump for 500 LPM and 10 meters head")
    st.write("• What pumps can handle 200 LPM?")
    st.write("• Show me pumps with 15m head capacity")
    st.write("• What's the best pump for my pool?")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your intelligent pump selection assistant. I can help you find the perfect pump based on your flow and head requirements. Just tell me what you need! 🚀"}
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me about pumps... (e.g., 'I need 500 LPM and 10m head')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if df.empty:
            assistant_response = "I'm sorry, but I can't access the pump database right now. Please check the connection and try again."
        else:
            # Extract flow and head from user message
            flow, head = extract_flow_head_from_text(prompt)
            
            # Generate AI response
            assistant_response = generate_ai_response(prompt, df, flow, head)
        
        # Simulate stream of response with typing effect
        words = assistant_response.split()
        for i, word in enumerate(words):
            full_response += word + " "
            if i % 3 == 0:  # Update every 3 words for smoother animation
                time.sleep(0.03)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        
        # If we found specific pumps, show them in a table
        if not df.empty and flow is not None and head is not None:
            filtered_pumps = search_pumps(df, flow, head)
            if not filtered_pumps.empty:
                st.subheader("🎯 Matching Pumps")
                display_columns = ["Model No.", "Max Flow (LPM)", "Max Head (M)"]
                if "Product Link" in filtered_pumps.columns:
                    display_columns.append("Product Link")
                
                st.dataframe(
                    filtered_pumps[display_columns].head(10).reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True
                )
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear chat history button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your intelligent pump selection assistant. I can help you find the perfect pump based on your flow and head requirements. Just tell me what you need! 🚀"}
    ]
    st.rerun()
