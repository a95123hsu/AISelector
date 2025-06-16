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
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

# --- RAG Setup with LangChain ---
@st.cache_resource
def setup_langchain_rag(df):
    """Setup LangChain RAG pipeline with pump data"""
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OpenAI API key required for RAG functionality")
        return None
    
    try:
        # Convert DataFrame to LangChain documents
        documents = []
        
        for _, row in df.iterrows():
            # Create comprehensive pump document
            pump_specs = [
                f"Model: {row['Model No.']}",
                f"Maximum Flow: {row['Max Flow (LPM)']} LPM",
                f"Maximum Head: {row['Max Head (M)']} meters"
            ]
            
            # Add performance curve data
            performance_data = []
            for col in df.columns:
                if col.endswith('M') and col not in ['Model No.', 'Max Flow (LPM)', 'Max Head (M)', 'Max Head(M)']:
                    if pd.notna(row[col]) and row[col] > 0:
                        head_val = col.replace('M', '').strip()
                        performance_data.append(f"At {head_val}m head: {row[col]} LPM flow")
            
            if performance_data:
                pump_specs.append("Performance curve:")
                pump_specs.extend(performance_data)
            
            # Add other specifications
            for col in df.columns:
                if col not in ['Model No.', 'Max Flow (LPM)', 'Max Head (M)', 'Max Head(M)'] and not col.endswith('M'):
                    if pd.notna(row[col]) and str(row[col]).strip():
                        pump_specs.append(f"{col}: {row[col]}")
            
            # Create document
            document_text = "\n".join(pump_specs)
            doc = Document(
                page_content=document_text,
                metadata={
                    "model": row['Model No.'],
                    "max_flow": row['Max Flow (LPM)'],
                    "max_head": row['Max Head (M)']
                }
            )
            documents.append(doc)
        
        # Split documents if needed
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        # Create retrieval QA chain
        llm = OpenAI(
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            temperature=0.3,
            max_tokens=200
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
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
            return None
        
        pump_row = pump_data.iloc[0]
        
        # Extract head vs flow data points
        flows = []
        heads = []
        
        # Get all columns that represent head measurements
        for col in df.columns:
            if col.endswith('M') and col not in ['Model No.', 'Max Flow (LPM)', 'Max Head (M)', 'Max Head(M)']:
                try:
                    # Extract head value from column name
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
            name=f'{model_no} Performance Curve',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8, color='#2E86AB'),
            hovertemplate='Flow: %{x} LPM<br>Head: %{y} m<extra></extra>'
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
            hovermode='closest'
        )
        
        # Add grid and styling
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting pump curve: {str(e)}")
        return None

# --- Helper Functions ---
def extract_requirements_from_text(text):
    """Extract flow, head, and model requirements from user text"""
    # Flow patterns
    flow_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:lpm|l/min|liters?\s*per\s*minute)',
        r'flow[:\s]*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*flow'
    ]
    
    # Head patterns
    head_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?\s*(?:head|height)',
        r'head[:\s]*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?'
    ]
    
    # Model patterns
    model_patterns = [
        r'(?:show|plot|curve|graph).*?([A-Z0-9]+[A-Z]+[0-9]+(?:\.[0-9]+)?)',
        r'model[:\s]*([A-Z0-9]+[A-Z]+[0-9]+(?:\.[0-9]+)?)',
        r'([A-Z0-9]+[A-Z]+[0-9]+(?:\.[0-9]+)?).*?(?:curve|plot|graph)'
    ]
    
    flow = None
    head = None
    model_no = None
    
    text_lower = text.lower()
    text_upper = text.upper()
    
    # Extract flow
    for pattern in flow_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            flow = float(match.group(1))
            break
    
    # Extract head
    for pattern in head_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            head = float(match.group(1))
            break
    
    # Extract model
    for pattern in model_patterns:
        match = re.search(pattern, text_upper, re.IGNORECASE)
        if match:
            model_no = match.group(1)
            break
    
    return flow, head, model_no

def search_pumps_by_requirements(df, flow=None, head=None):
    """Search for suitable pumps based on requirements"""
    if df.empty:
        return pd.DataFrame()
    
    filtered = df.copy()
    
    if flow is not None:
        filtered = filtered[filtered["Max Flow (LPM)"] >= flow]
    
    if head is not None:
        filtered = filtered[filtered["Max Head (M)"] >= head]
    
    if not filtered.empty and flow is not None and head is not None:
        # Calculate efficiency scores
        filtered = filtered.copy()
        filtered["flow_efficiency"] = filtered["Max Flow (LPM)"] / flow
        filtered["head_efficiency"] = filtered["Max Head (M)"] / head
        filtered["efficiency_score"] = filtered["flow_efficiency"] + filtered["head_efficiency"]
        filtered = filtered.sort_values("efficiency_score")
    
    return filtered

def generate_rag_response(user_message, qa_chain, df, flow=None, head=None, model_no=None):
    """Generate response using LangChain RAG"""
    try:
        # Handle pump curve requests
        if model_no or any(word in user_message.lower() for word in ['curve', 'plot', 'graph', 'chart']):
            if not model_no:
                flow, head, model_no = extract_requirements_from_text(user_message)
            
            if model_no and model_no in df["Model No."].values:
                return {
                    "response": f"Here's the performance curve for {model_no}.",
                    "show_pumps": False,
                    "show_curve": True,
                    "pumps_data": pd.DataFrame(),
                    "curve_model": model_no
                }
        
        # Create enhanced query for RAG
        enhanced_query = f"""
User request: {user_message}

Requirements:
- Flow needed: {flow if flow else 'not specified'} LPM
- Head needed: {head if head else 'not specified'} meters

Please recommend suitable pumps based on these requirements. 
Focus on models that can meet or exceed the specifications.
Keep the response concise (2-3 sentences maximum).
"""
        
        # Use RAG chain
        if qa_chain:
            result = qa_chain({"query": enhanced_query})
            response_text = result["answer"]
            
            # Get source models for additional context
            source_models = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    if "model" in doc.metadata:
                        source_models.append(doc.metadata["model"])
        else:
            response_text = "RAG system not available. Using basic search."
        
        # Get matching pumps for table display
        filtered_pumps = search_pumps_by_requirements(df, flow, head)
        
        return {
            "response": response_text,
            "show_pumps": len(filtered_pumps) > 0,
            "show_curve": False,
            "pumps_data": filtered_pumps,
            "curve_model": None
        }
        
    except Exception as e:
        # Fallback to basic search
        filtered_pumps = search_pumps_by_requirements(df, flow, head)
        
        if not filtered_pumps.empty:
            response_text = f"Found {len(filtered_pumps)} pumps that meet your requirements."
        else:
            response_text = "No pumps found for your specific requirements. Try adjusting your specifications."
        
        return {
            "response": response_text,
            "show_pumps": len(filtered_pumps) > 0,
            "show_curve": False,
            "pumps_data": filtered_pumps,
            "curve_model": None
        }

# --- Streamlit App ---
st.set_page_config(
    page_title="AI Pump Selector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI-Powered Pump Selector")
st.caption("Advanced pump recommendations using LangChain RAG technology 🤖")

# Sidebar configuration
with st.sidebar:
    st.header("📊 Database Settings")
    
    # Table selection
    selected_table = st.selectbox(
        "Select Pump Data Table",
        ["pump_curve_data", "pump_selection_data"],
        help="Choose which table to use for pump data"
    )
    
    # Load data
    with st.spinner("Loading pump data..."):
        df = load_data(selected_table)
    
    if not df.empty:
        st.success(f"✅ {len(df)} pumps loaded")
        st.metric("Flow Range", f"{df['Max Flow (LPM)'].min():.0f} - {df['Max Flow (LPM)'].max():.0f} LPM")
        st.metric("Head Range", f"{df['Max Head (M)'].min():.1f} - {df['Max Head (M)'].max():.1f} m")
        
        # Setup RAG pipeline
        if "OPENAI_API_KEY" in st.secrets:
            with st.spinner("Initializing AI knowledge base..."):
                qa_chain = setup_langchain_rag(df)
            
            if qa_chain:
                st.success("🧠 LangChain RAG ready!")
            else:
                st.error("❌ RAG setup failed")
                qa_chain = None
        else:
            st.warning("⚠️ Add OpenAI API key for RAG functionality")
            qa_chain = None
        
        # Show available models
        st.header("🎯 Available Models")
        models = df["Model No."].unique()[:8]
        for model in models:
            st.text(f"• {model}")
        if len(df) > 8:
            st.text(f"... and {len(df) - 8} more")
            
    else:
        st.error("❌ No data loaded")
        qa_chain = None
    
    st.header("💡 Example Queries")
    st.write("• I need 80 LPM and 10 meters head")
    st.write("• Best pump for swimming pool")
    st.write("• Show curve for 65ADL51.5")
    st.write("• High efficiency pumps")
    st.write("• Industrial applications")

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! I'm your AI pump selection assistant powered by LangChain RAG. I can help you find the perfect pump and show performance curves. What are your requirements? 🚀"
        }
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about pumps... (e.g., 'I need 80 LPM and 10 meters head')"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if df.empty:
            response_data = {
                "response": "I can't access the pump database. Please check your connection.",
                "show_pumps": False,
                "show_curve": False,
                "pumps_data": pd.DataFrame(),
                "curve_model": None
            }
        else:
            # Extract requirements
            flow, head, model_no = extract_requirements_from_text(prompt)
            
            # Generate response using RAG
            response_data = generate_rag_response(prompt, qa_chain, df, flow, head, model_no)
        
        # Typing animation
        full_response = ""
        assistant_response = response_data["response"]
        
        for i, word in enumerate(assistant_response.split()):
            full_response += word + " "
            if i % 3 == 0:
                time.sleep(0.02)
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        
        # Show pump curve if requested
        if response_data["show_curve"] and response_data["curve_model"]:
            st.subheader(f"📊 Performance Curve - {response_data['curve_model']}")
            curve_fig = plot_pump_curve(df, response_data["curve_model"])
            if curve_fig:
                st.plotly_chart(curve_fig, use_container_width=True)
            else:
                st.warning(f"Insufficient data to plot curve for {response_data['curve_model']}")
        
        # Show recommended pumps
        if response_data["show_pumps"] and not response_data["pumps_data"].empty:
            st.subheader("🎯 Recommended Pumps")
            
            # Display full dataframe
            st.dataframe(
                response_data["pumps_data"].head(10).reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )
            
            # Quick curve access
            if len(response_data["pumps_data"]) > 0:
                st.write("**View pump curves:**")
                cols = st.columns(min(4, len(response_data["pumps_data"])))
                
                for i, (_, pump) in enumerate(response_data["pumps_data"].head(4).iterrows()):
                    with cols[i]:
                        if st.button(f"📊 {pump['Model No.']}", key=f"curve_btn_{i}"):
                            curve_fig = plot_pump_curve(df, pump['Model No.'])
                            if curve_fig:
                                st.plotly_chart(curve_fig, use_container_width=True)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! I'm your AI pump selection assistant powered by LangChain RAG. I can help you find the perfect pump and show performance curves. What are your requirements? 🚀"
        }
    ]
    st.rerun()
