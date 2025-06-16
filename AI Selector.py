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
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="AI Pump Selector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize OpenAI Client ---
@st.cache_resource
def init_openai_client():
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("🔑 OpenAI API key not found in secrets!")
        return None
    return openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Initialize Supabase ---
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        supabase = create_client(url, key)
        return supabase
    except Exception as e:
        st.error(f"💾 Error connecting to Supabase: {str(e)}")
        return None

# --- Load pump data ---
@st.cache_data
def load_pump_data(table_name):
    """Load and clean pump data from Supabase"""
    supabase = init_supabase()
    if not supabase:
        return pd.DataFrame()
    
    try:
        response = supabase.table(table_name).select("*").execute()
        df = pd.DataFrame(response.data)
        
        if df.empty:
            return df
            
        # Clean column names
        if "Max Head(M)" in df.columns and "Max Head (M)" not in df.columns:
            df["Max Head (M)"] = df["Max Head(M)"]
        
        # Validate required columns
        required_cols = ["Model No.", "Max Flow (LPM)", "Max Head (M)"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            return pd.DataFrame()
        
        # Clean numeric columns
        for col in ["Max Flow (LPM)", "Max Head (M)"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Remove invalid rows
        df = df.dropna(subset=["Max Flow (LPM)", "Max Head (M)"])
        
        return df
        
    except Exception as e:
        st.error(f"💾 Error loading data: {str(e)}")
        return pd.DataFrame()

# --- Setup RAG system ---
@st.cache_resource
def setup_rag_system(_df):
    """Initialize LangChain RAG system"""
    if _df.empty or "OPENAI_API_KEY" not in st.secrets:
        return None
    
    try:
        # Create documents from pump data
        documents = []
        for _, pump in _df.iterrows():
            # Basic pump info
            content = f"""
Model: {pump['Model No.']}
Maximum Flow: {pump['Max Flow (LPM)']} LPM
Maximum Head: {pump['Max Head (M)']} meters
"""
            
            # Add performance curve data
            curve_points = []
            for col in _df.columns:
                if col.endswith('M') and col not in ['Model No.', 'Max Flow (LPM)', 'Max Head (M)', 'Max Head(M)']:
                    if pd.notna(pump[col]) and pump[col] > 0:
                        head = col.replace('M', '').strip()
                        curve_points.append(f"At {head}m head: {pump[col]} LPM")
            
            if curve_points:
                content += "\nPerformance Data:\n" + "\n".join(curve_points)
            
            # Add other specifications
            for col in _df.columns:
                if col not in ['Model No.', 'Max Flow (LPM)', 'Max Head (M)', 'Max Head(M)'] and not col.endswith('M'):
                    if pd.notna(pump[col]) and str(pump[col]).strip():
                        content += f"\n{col}: {pump[col]}"
            
            doc = Document(
                page_content=content.strip(),
                metadata={"model": pump['Model No.']}
            )
            documents.append(doc)
        
        # Create vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Create retrieval chain
        llm = ChatOpenAI(
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            model="gpt-4o-mini",
            temperature=0.3
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"🧠 RAG setup failed: {str(e)}")
        return None

# --- Helper functions ---
def extract_pump_requirements(text):
    """Extract flow, head, and model from user input"""
    # Flow patterns
    flow_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lpm|l/min|flow)', text.lower())
    flow = float(flow_match.group(1)) if flow_match else None
    
    # Head patterns  
    head_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meter|head)', text.lower())
    head = float(head_match.group(1)) if head_match else None
    
    # Model patterns
    model_match = re.search(r'([A-Z0-9]+[A-Z]+[0-9]+(?:\.[0-9]+)?)', text.upper())
    model = model_match.group(1) if model_match else None
    
    return flow, head, model

def filter_pumps(df, flow=None, head=None):
    """Filter pumps based on requirements"""
    if df.empty:
        return df
    
    filtered = df.copy()
    
    if flow:
        filtered = filtered[filtered["Max Flow (LPM)"] >= flow]
    if head:
        filtered = filtered[filtered["Max Head (M)"] >= head]
    
    # Sort by efficiency if both requirements specified
    if flow and head and not filtered.empty:
        filtered["efficiency"] = (filtered["Max Flow (LPM)"] / flow) + (filtered["Max Head (M)"] / head)
        filtered = filtered.sort_values("efficiency")
    
    return filtered

def create_pump_curve(df, model_no):
    """Generate pump performance curve"""
    pump_data = df[df["Model No."] == model_no]
    if pump_data.empty:
        return None
    
    pump = pump_data.iloc[0]
    
    # Extract curve data
    heads, flows = [], []
    for col in df.columns:
        if col.endswith('M') and col not in ['Model No.', 'Max Flow (LPM)', 'Max Head (M)', 'Max Head(M)']:
            try:
                head_val = float(col.replace('M', '').strip())
                flow_val = pump[col]
                if pd.notna(flow_val) and flow_val > 0:
                    heads.append(head_val)
                    flows.append(float(flow_val))
            except ValueError:
                continue
    
    if len(flows) < 2:
        return None
    
    # Sort data
    sorted_data = sorted(zip(heads, flows))
    heads, flows = zip(*sorted_data)
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=flows, y=heads,
        mode='lines+markers',
        name=f'{model_no}',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8),
        hovertemplate='Flow: %{x} LPM<br>Head: %{y}m<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Pump Curve - {model_no}',
        xaxis_title='Flow (LPM)',
        yaxis_title='Head (m)',
        template='plotly_white',
        height=400
    )
    
    return fig

# --- Streamlit Chat Implementation ---
def main():
    # App header
    st.title("🔍 AI Pump Selector")
    st.caption("Powered by LangChain RAG + OpenAI")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Table selection
        table_name = st.selectbox(
            "Select Data Source",
            ["pump_curve_data", "pump_selection_data"],
            help="Choose your pump database table"
        )
        
        # Load data
        with st.spinner("Loading pump data..."):
            df = load_pump_data(table_name)
        
        if not df.empty:
            st.success(f"✅ {len(df)} pumps loaded")
            
            # Setup RAG
            with st.spinner("Setting up AI..."):
                qa_chain = setup_rag_system(df)
            
            if qa_chain:
                st.success("🧠 AI Ready!")
            else:
                st.warning("⚠️ AI unavailable")
                qa_chain = None
                
            # Display stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Flow Range", f"{df['Max Flow (LPM)'].min():.0f}-{df['Max Flow (LPM)'].max():.0f}")
            with col2:
                st.metric("Head Range", f"{df['Max Head (M)'].min():.1f}-{df['Max Head (M)'].max():.1f}")
            
            # Show sample models
            st.subheader("📋 Available Models")
            for model in df["Model No."].head(6):
                st.code(model, language=None)
                
        else:
            st.error("❌ No data available")
            qa_chain = None
        
        st.divider()
        
        # Example queries
        st.subheader("💡 Try These")
        examples = [
            "I need 100 LPM and 8m head",
            "Best pump for swimming pool",
            "Show curve for 65ADL51.5",
            "High flow industrial pump"
        ]
        
        for example in examples:
            if st.button(example, key=f"ex_{example[:10]}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Welcome message
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.write("👋 Hi! I'm your AI pump expert. I can help you:")
            st.write("• Find pumps for specific flow/head requirements")
            st.write("• Show performance curves for any model")
            st.write("• Recommend pumps for different applications")
            st.write("\nWhat can I help you with today?")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about pumps..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            if df.empty:
                st.error("No pump data available. Please check your database connection.")
                return
            
            # Extract requirements
            flow, head, model = extract_pump_requirements(prompt)
            
            # Handle curve requests
            if model and any(word in prompt.lower() for word in ['curve', 'plot', 'graph']):
                if model in df["Model No."].values:
                    st.write(f"Here's the performance curve for **{model}**:")
                    fig = create_pump_curve(df, model)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        response = f"Displayed pump curve for {model}"
                    else:
                        st.warning("Insufficient data for curve generation")
                        response = f"Cannot generate curve for {model} - insufficient data"
                else:
                    st.error(f"Model {model} not found in database")
                    response = f"Model {model} not found"
            
            # Handle pump search requests
            else:
                # Use RAG if available
                if qa_chain:
                    with st.spinner("🧠 Thinking..."):
                        try:
                            result = qa_chain({"query": prompt})
                            response = result["answer"]
                            st.write(response)
                            
                            # Show related pumps if requirements found
                            if flow or head:
                                filtered = filter_pumps(df, flow, head)
                                if not filtered.empty:
                                    st.subheader("🎯 Recommended Pumps")
                                    st.dataframe(
                                        filtered[["Model No.", "Max Flow (LPM)", "Max Head (M)"]].head(5),
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                    
                                    # Quick curve buttons
                                    st.write("**View curves:**")
                                    cols = st.columns(min(3, len(filtered)))
                                    for i, (_, pump) in enumerate(filtered.head(3).iterrows()):
                                        with cols[i]:
                                            if st.button(f"📊 {pump['Model No.']}", key=f"curve_{i}"):
                                                fig = create_pump_curve(df, pump['Model No.'])
                                                if fig:
                                                    st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"AI error: {str(e)}")
                            response = "Sorry, I encountered an error. Please try rephrasing your question."
                            st.write(response)
                
                # Fallback to basic search
                else:
                    if flow or head:
                        filtered = filter_pumps(df, flow, head)
                        if not filtered.empty:
                            response = f"Found {len(filtered)} pumps matching your requirements."
                            st.write(response)
                            st.dataframe(
                                filtered[["Model No.", "Max Flow (LPM)", "Max Head (M)"]].head(5),
                                use_container_width=True
                            )
                        else:
                            response = "No pumps found matching your requirements."
                            st.write(response)
                    else:
                        response = "Please specify your flow (LPM) and head (m) requirements."
                        st.write(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
