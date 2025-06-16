import streamlit as st
import pandas as pd
import openai
from supabase import create_client
import time
from postgrest.exceptions import APIError
import re
import plotly.graph_objects as go
import plotly.express as px

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
            if "Max Head(M)" in df.columns and "Max Head (M)" not in df.columns:
                df["Max Head (M)"] = df["Max Head(M)"]
            
            # Ensure we have the required columns
            required_columns = ["Model No.", "Max Flow (LPM)"]
            head_column = "Max Head(M)" if "Max Head(M)" in df.columns else "Max Head (M)"
            
            if head_column in df.columns:
                required_columns.append(head_column)
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

# --- Pump Curve Plotting ---
def plot_pump_curve(df, model_no):
    """Generate pump curve plot for a specific model"""
    try:
        # Find the pump model
        pump_data = df[df["Model No."] == model_no]
        if pump_data.empty:
            return None
        
        pump_row = pump_data.iloc[0]
        
        # Extract head vs flow data points
        head_columns = [col for col in df.columns if 'M' in col and col not in ['Model No.', 'Max Flow (LPM)', 'Max Head (M)']]
        head_columns = [col for col in head_columns if col.replace('M', '').replace('.', '').replace(' ', '').isdigit()]
        
        flows = []
        heads = []
        
        for col in head_columns:
            if pd.notna(pump_row[col]) and pump_row[col] > 0:
                # Extract head value from column name
                head_str = col.replace('M', '').replace(' ', '')
                try:
                    head_val = float(head_str)
                    flow_val = float(pump_row[col])
                    heads.append(head_val)
                    flows.append(flow_val)
                except:
                    continue
        
        if len(flows) < 2:
            return None
        
        # Sort by head for proper curve
        sorted_data = sorted(zip(heads, flows))
        heads, flows = zip(*sorted_data)
        
        # Create the plot
        fig = go.Figure()
        
        # Add pump curve
        fig.add_trace(go.Scatter(
            x=flows,
            y=heads,
            mode='lines+markers',
            name=f'{model_no} Pump Curve',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, color='#1f77b4')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Pump Performance Curve - {model_no}',
            xaxis_title='Flow (LPM)',
            yaxis_title='Head (meters)',
            showlegend=True,
            template='plotly_white',
            width=600,
            height=400,
            hovermode='x unified'
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting pump curve: {str(e)}")
        return None

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

def extract_model_from_text(text):
    """Extract pump model number from user text"""
    # Look for patterns like "show curve for 65ADL51.5" or "plot 80ADL52.2"
    model_patterns = [
        r'(?:show|plot|curve|graph).*?([A-Z0-9]+[A-Z]+[0-9]+(?:\.[0-9]+)?)',
        r'model[:\s]*([A-Z0-9]+[A-Z]+[0-9]+(?:\.[0-9]+)?)',
        r'([A-Z0-9]+[A-Z]+[0-9]+(?:\.[0-9]+)?).*?(?:curve|plot|graph)'
    ]
    
    text_upper = text.upper()
    
    for pattern in model_patterns:
        match = re.search(pattern, text_upper, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

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

def generate_organized_response(user_message, df, flow=None, head=None, model_no=None):
    """Generate organized AI response based on user message and pump data"""
    if "OPENAI_API_KEY" not in st.secrets:
        return {
            "response": "I'd love to help you with pump selection, but I need an OpenAI API key to provide intelligent responses. For now, I can help you search for pumps if you tell me your flow (LPM) and head (meters) requirements!",
            "show_pumps": False,
            "show_curve": False,
            "pumps_data": pd.DataFrame(),
            "curve_model": None
        }
    
    try:
        # Check if user wants to see a pump curve
        if model_no or any(word in user_message.lower() for word in ['curve', 'plot', 'graph', 'chart']):
            if not model_no:
                model_no = extract_model_from_text(user_message)
            
            if model_no and model_no in df["Model No."].values:
                context = f"""
You are a pump expert. The user asked about pump model {model_no}. 

Provide a brief, organized response that:
1. Acknowledges their request for the pump curve
2. Mentions key specifications of this model
3. Explains what they'll see in the curve plot

Keep it concise and professional. The pump curve will be displayed automatically.
"""
                
                client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": context}],
                    max_tokens=200,
                    temperature=0.7
                )
                
                return {
                    "response": response.choices[0].message.content,
                    "show_pumps": False,
                    "show_curve": True,
                    "pumps_data": pd.DataFrame(),
                    "curve_model": model_no
                }
        
        # Search for relevant pumps
        filtered_pumps = search_pumps(df, flow, head)
        
        # Prepare context for AI
        context = f"""
You are a pump selection expert. Provide a well-organized response to the user.

User message: "{user_message}"

Database info:
- Total pumps available: {len(df)}
- Flow range: {df['Max Flow (LPM)'].min():.0f} - {df['Max Flow (LPM)'].max():.0f} LPM
- Head range: {df['Max Head (M)'].min():.1f} - {df['Max Head (M)'].max():.1f} meters

"""

        if flow is not None or head is not None:
            context += f"""
Requirements extracted:
- Flow needed: {flow if flow else 'Not specified'} LPM
- Head needed: {head if head else 'Not specified'} meters

"""

        if not filtered_pumps.empty:
            top_5 = filtered_pumps.head(5)
            context += f"""
Found {len(filtered_pumps)} suitable pumps. Top 5 matches:
{top_5[['Model No.', 'Max Flow (LPM)', 'Max Head (M)']].to_string(index=False)}
"""
        else:
            if flow is not None and head is not None:
                context += f"No pumps found for {flow} LPM and {head}m. Suggest alternatives."

        context += """

Provide a response with this structure:
1. **Summary**: Brief acknowledgment of their request
2. **Analysis**: What you found based on their requirements
3. **Recommendations**: Specific pump suggestions (if any)
4. **Next Steps**: What they can do next or questions to ask

Use markdown formatting for headers. Be concise but helpful.
"""

        # Call OpenAI API
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": context}],
            max_tokens=400,
            temperature=0.7
        )
        
        return {
            "response": response.choices[0].message.content,
            "show_pumps": len(filtered_pumps) > 0,
            "show_curve": False,
            "pumps_data": filtered_pumps,
            "curve_model": None
        }
        
    except Exception as e:
        return {
            "response": f"I encountered an error: {str(e)}. But I'm still here to help! Could you tell me your flow (LPM) and head (meters) requirements?",
            "show_pumps": False,
            "show_curve": False,
            "pumps_data": pd.DataFrame(),
            "curve_model": None
        }

# --- App UI ---
st.title("🔍 Pump Selector Assistant")
st.caption("Ask me about pump requirements and I'll help you find the perfect match! Try asking for pump curves too! 📊")

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
        
        st.header("🎯 Available Models")
        models = df["Model No."].unique()[:10]  # Show first 10 models
        for model in models:
            st.text(f"• {model}")
        if len(df) > 10:
            st.text(f"... and {len(df) - 10} more")
    else:
        st.error("❌ No data loaded")
    
    st.header("💡 Example Questions")
    st.write("• I need 500 LPM and 10 meters head")
    st.write("• Show me pumps for high flow")
    st.write("• Plot curve for model 65ADL51.5")
    st.write("• What's the best pump for 300 LPM?")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your intelligent pump selection assistant. I can help you:\n\n🔍 **Find pumps** based on your flow and head requirements\n📊 **Show pump curves** for specific models\n💡 **Provide recommendations** for your applications\n\nJust tell me what you need!"}
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me about pumps... (e.g., 'I need 500 LPM and 10m head' or 'show curve for 65ADL51.5')"):
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
            response_data = {
                "response": assistant_response,
                "show_pumps": False,
                "show_curve": False,
                "pumps_data": pd.DataFrame(),
                "curve_model": None
            }
        else:
            # Extract requirements from user message
            flow, head = extract_flow_head_from_text(prompt)
            model_no = extract_model_from_text(prompt)
            
            # Generate organized AI response
            response_data = generate_organized_response(prompt, df, flow, head, model_no)
        
        assistant_response = response_data["response"]
        
        # Simulate stream of response with typing effect
        words = assistant_response.split()
        for i, word in enumerate(words):
            full_response += word + " "
            if i % 3 == 0:  # Update every 3 words for smoother animation
                time.sleep(0.02)
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        
        # Show pump curve if requested
        if response_data["show_curve"] and response_data["curve_model"]:
            st.subheader(f"📊 Pump Curve - {response_data['curve_model']}")
            curve_fig = plot_pump_curve(df, response_data["curve_model"])
            if curve_fig:
                st.plotly_chart(curve_fig, use_container_width=True)
            else:
                st.warning(f"Could not generate pump curve for {response_data['curve_model']}. Insufficient data points.")
        
        # Show matching pumps table if found
        if response_data["show_pumps"] and not response_data["pumps_data"].empty:
            st.subheader("🎯 Recommended Pumps")
            display_columns = ["Model No.", "Max Flow (LPM)", "Max Head (M)"]
            if "Product Link" in response_data["pumps_data"].columns:
                display_columns.append("Product Link")
            
            st.dataframe(
                response_data["pumps_data"][display_columns].head(10).reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )
            
            # Add buttons to show curves for recommended pumps
            if len(response_data["pumps_data"]) > 0:
                st.write("**Show pump curves:**")
                cols = st.columns(min(5, len(response_data["pumps_data"])))
                for i, (_, pump) in enumerate(response_data["pumps_data"].head(5).iterrows()):
                    with cols[i]:
                        if st.button(f"📊 {pump['Model No.']}", key=f"curve_{i}"):
                            curve_fig = plot_pump_curve(df, pump['Model No.'])
                            if curve_fig:
                                st.plotly_chart(curve_fig, use_container_width=True)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear chat history button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your intelligent pump selection assistant. I can help you:\n\n🔍 **Find pumps** based on your flow and head requirements\n📊 **Show pump curves** for specific models\n💡 **Provide recommendations** for your applications\n\nJust tell me what you need!"}
    ]
    st.rerun()
