import streamlit as st
import pandas as pd
import openai
from supabase import create_client
import time
from postgrest.exceptions import APIError
import re
import plotly.graph_objects as go
import io

# LangChain imports
try:
    from langchain_openai import OpenAIEmbeddings, OpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.docstore.document import Document
    from langchain.text_splitter import CharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("⚠️ LangChain not installed. Install with: pip install langchain langchain-openai langchain-community faiss-cpu")

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

# --- RAG Setup ---
@st.cache_resource
def setup_rag_pipeline(df):
    """Setup LangChain RAG pipeline with pump data"""
    if not LANGCHAIN_AVAILABLE or "OPENAI_API_KEY" not in st.secrets:
        return None
    
    try:
        # Convert DataFrame to documents
        documents = []
        for _, row in df.iterrows():
            # Create a comprehensive text representation of each pump
            pump_text = f"""
Model: {row['Model No.']}
Max Flow: {row['Max Flow (LPM)']} LPM
Max Head: {row['Max Head (M)']} meters
"""
            
            # Add performance data points if available
            performance_points = []
            for col in df.columns:
                if col.endswith('M') and col not in ['Model No.', 'Max Flow (LPM)', 'Max Head (M)', 'Max Head(M)']:
                    if pd.notna(row[col]) and row[col] > 0:
                        head_val = col.replace('M', '').strip()
                        performance_points.append(f"At {head_val}m head: {row[col]} LPM")
            
            if performance_points:
                pump_text += "Performance curve data:\n" + "\n".join(performance_points)
            
            # Add other specifications if available
            for col in df.columns:
                if col not in ['Model No.', 'Max Flow (LPM)', 'Max Head (M)', 'Max Head(M)'] and not col.endswith('M'):
                    if pd.notna(row[col]):
                        pump_text += f"\n{col}: {row[col]}"
            
            doc = Document(page_content=pump_text, metadata={"model": row['Model No.']})
            documents.append(doc)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Create RAG chain
        llm = OpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Error setting up RAG pipeline: {str(e)}")
        return None

# --- Pump Curve Plotting ---
def plot_pump_curve(df, model_no):
    """Generate pump curve plot for a specific model"""
    try:
        # Find the pump model
        pump_data = df[df["Model No."] == model_no]
        if pump_data.empty:
            st.error(f"Model {model_no} not found in database")
            return None
        
        pump_row = pump_data.iloc[0]
        
        # Extract head vs flow data points from columns like "3M", "6M", "9M" etc.
        flows = []
        heads = []
        
        # Get all columns that represent head measurements
        for col in df.columns:
            if col.endswith('M') and col not in ['Model No.', 'Max Flow (LPM)', 'Max Head (M)', 'Max Head(M)']:
                try:
                    # Extract head value from column name (e.g., "3M" -> 3.0)
                    head_str = col.replace('M', '').strip()
                    if head_str.replace('.', '').isdigit():
                        head_val = float(head_str)
                        flow_val = pump_row[col]
                        
                        # Only add if flow value exists and is positive
                        if pd.notna(flow_val) and flow_val > 0:
                            heads.append(head_val)
                            flows.append(float(flow_val))
                except (ValueError, TypeError):
                    continue
        
        if len(flows) < 2:
            st.warning(f"Not enough data points to plot curve for {model_no} (found {len(flows)} points)")
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
            width=700,
            height=500,
            hovermode='x unified'
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting pump curve: {str(e)}")
        return None

# --- Helper Functions ---
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

def generate_rag_response(user_message, qa_chain, df, flow=None, head=None, model_no=None):
    """Generate response using RAG pipeline"""
    try:
        # Check if user wants to see a pump curve
        if model_no or any(word in user_message.lower() for word in ['curve', 'plot', 'graph', 'chart']):
            if not model_no:
                model_no = extract_model_from_text(user_message)
            
            if model_no and model_no in df["Model No."].values:
                return {
                    "response": f"Here's the pump curve for {model_no}. You can see its flow vs head performance below.",
                    "show_pumps": False,
                    "show_curve": True,
                    "pumps_data": pd.DataFrame(),
                    "curve_model": model_no
                }
        
        # Use RAG for pump selection
        if qa_chain:
            # Enhance the question with context
            enhanced_query = f"""
User question: {user_message}

Please help find suitable pumps. Consider:
- Flow requirement: {flow if flow else 'not specified'} LPM
- Head requirement: {head if head else 'not specified'} meters
- Provide specific model recommendations with brief explanations
- Keep response concise (2-3 sentences max)
"""
            
            result = qa_chain({"query": enhanced_query})
            response_text = result["answer"]
            
            # Get matching pumps for table display
            filtered_pumps = search_pumps(df, flow, head)
            
            return {
                "response": response_text,
                "show_pumps": len(filtered_pumps) > 0,
                "show_curve": False,
                "pumps_data": filtered_pumps,
                "curve_model": None
            }
        else:
            # Fallback to simple search if RAG not available
            filtered_pumps = search_pumps(df, flow, head)
            
            if not filtered_pumps.empty:
                response_text = f"Found {len(filtered_pumps)} pumps that meet your requirements. Check the recommendations below."
            else:
                response_text = "No pumps found for your specific requirements. Try adjusting your flow or head needs."
            
            return {
                "response": response_text,
                "show_pumps": len(filtered_pumps) > 0,
                "show_curve": False,
                "pumps_data": filtered_pumps,
                "curve_model": None
            }
            
    except Exception as e:
        return {
            "response": f"Error processing your request: {str(e)}. Please try again.",
            "show_pumps": False,
            "show_curve": False,
            "pumps_data": pd.DataFrame(),
            "curve_model": None
        }

# --- App UI ---
st.title("🔍 AI-Powered Pump Selector")
st.caption("Smart pump recommendations using RAG (Retrieval-Augmented Generation) 🤖📊")

# Table selection in sidebar
with st.sidebar:
    st.header("📊 Database Settings")
    selected_table = st.selectbox(
        "Select Pump Data Table",
        ["pump_curve_data", "pump_selection_data"],
        help="Choose which table to use for pump data"
    )
    
    # Load data
    df = load_data(selected_table)
    
    if not df.empty:
        st.success(f"✅ {len(df)} pumps loaded")
        st.metric("Flow Range", f"{df['Max Flow (LPM)'].min():.0f} - {df['Max Flow (LPM)'].max():.0f} LPM")
        st.metric("Head Range", f"{df['Max Head (M)'].min():.1f} - {df['Max Head (M)'].max():.1f} m")
        
        # Setup RAG pipeline
        if LANGCHAIN_AVAILABLE and "OPENAI_API_KEY" in st.secrets:
            with st.spinner("Setting up AI knowledge base..."):
                qa_chain = setup_rag_pipeline(df)
            if qa_chain:
                st.success("🧠 AI knowledge base ready!")
            else:
                st.warning("⚠️ Using basic search (RAG setup failed)")
                qa_chain = None
        else:
            st.warning("⚠️ RAG not available - using basic search")
            qa_chain = None
        
        st.header("🎯 Available Models")
        models = df["Model No."].unique()[:8]
        for model in models:
            st.text(f"• {model}")
        if len(df) > 8:
            st.text(f"... and {len(df) - 8} more")
    else:
        st.error("❌ No data loaded")
        qa_chain = None
    
    st.header("💡 Example Questions")
    st.write("• I need 500 LPM and 10 meters head")
    st.write("• Best pump for swimming pool")
    st.write("• Show curve for 65ADL51.5")
    st.write("• High flow low head applications")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI pump selection assistant powered by advanced RAG technology. I can help you find the perfect pump and show performance curves. What do you need? 🚀"}
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me about pumps... (e.g., 'I need 80 LPM and 10 meters head')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if df.empty:
            response_data = {
                "response": "I can't access the pump database right now. Please check the connection.",
                "show_pumps": False,
                "show_curve": False,
                "pumps_data": pd.DataFrame(),
                "curve_model": None
            }
        else:
            # Extract requirements from user message
            flow, head = extract_flow_head_from_text(prompt)
            model_no = extract_model_from_text(prompt)
            
            # Generate AI response using RAG
            response_data = generate_rag_response(prompt, qa_chain, df, flow, head, model_no)
        
        assistant_response = response_data["response"]
        
        # Simulate typing effect
        words = assistant_response.split()
        for i, word in enumerate(words):
            full_response += word + " "
            if i % 3 == 0:
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
                st.warning(f"Could not generate pump curve for {response_data['curve_model']}")
        
        # Show matching pumps table
        if response_data["show_pumps"] and not response_data["pumps_data"].empty:
            st.subheader("🎯 Recommended Pumps")
            st.dataframe(
                response_data["pumps_data"].head(10).reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )
            
            # Quick curve buttons
            if len(response_data["pumps_data"]) > 0:
                st.write("**Quick curve view:**")
                cols = st.columns(min(4, len(response_data["pumps_data"])))
                for i, (_, pump) in enumerate(response_data["pumps_data"].head(4).iterrows()):
                    with cols[i]:
                        if st.button(f"📊 {pump['Model No.']}", key=f"curve_{i}"):
                            curve_fig = plot_pump_curve(df, pump['Model No.'])
                            if curve_fig:
                                st.plotly_chart(curve_fig, use_container_width=True)
    
    # Add to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear chat history
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI pump selection assistant powered by advanced RAG technology. I can help you find the perfect pump and show performance curves. What do you need? 🚀"}
    ]
    st.rerun()
