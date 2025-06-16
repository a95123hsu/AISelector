import streamlit as st
import pandas as pd
import openai
from supabase import create_client
import time
from postgrest.exceptions import APIError

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
def load_data():
    if not supabase:
        return pd.DataFrame()
    
    try:
        response = supabase.table("pump_specs").select("*").execute()
        df = pd.DataFrame(response.data)
        
        # Clean and prepare the data
        if not df.empty:
            # Fix column name inconsistency (your code uses "Max Head (M)" but CSV has "Max Head(M)")
            if "Max Head(M)" in df.columns and "Max Head (M)" not in df.columns:
                df["Max Head (M)"] = df["Max Head(M)"]
            
            # Ensure numeric columns are properly typed
            numeric_columns = ["Max Flow (LPM)", "Max Head (M)"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Remove rows with NaN values in critical columns
            df = df.dropna(subset=numeric_columns)
        
        return df
    except APIError as e:
        st.error(f"Error accessing pump_specs table: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error loading data: {str(e)}")
        return pd.DataFrame()

# --- App UI ---
st.title("🔍 Pump Selector Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your pump selection assistant. I can help you find the right pump based on your flow and head requirements. What specifications do you need?"}
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input ---
col1, col2 = st.columns(2)
with col1:
    flow = st.number_input("Required Flow (LPM)", min_value=0.0, value=0.0, step=1.0)
with col2:
    head = st.number_input("Required Head (m)", min_value=0.0, value=0.0, step=0.5)

# Load data
df = load_data()

# --- Search and Filter ---
if st.button("Search", disabled=(flow <= 0 or head <= 0)):
    if flow <= 0 or head <= 0:
        st.warning("Please enter valid flow and head requirements.")
    else:
        # Add user message to chat history
        user_message = f"I need a pump with {flow} LPM flow and {head} meters head."
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_message)

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            if df.empty:
                response_text = "I'm sorry, but I'm having trouble accessing the pump database at the moment. Please try again later."
            else:
                # Filter pumps that meet requirements
                # Use >= for flow and head to find pumps that can handle AT LEAST the required specs
                filtered = df[
                    (df["Max Flow (LPM)"] >= flow) &
                    (df["Max Head (M)"] >= head)
                ]

                # Prepare response
                if filtered.empty:
                    response_text = f"I couldn't find any pumps that can handle {flow} LPM flow and {head} meters head. You might need to consider:\n\n"
                    response_text += "• Reducing your flow or head requirements\n"
                    response_text += "• Using multiple pumps in parallel (for higher flow)\n"
                    response_text += "• Using pumps in series (for higher head)\n"
                    response_text += "\nWould you like to try different specifications?"
                else:
                    # Sort by efficiency - closest to requirements without being oversized
                    filtered = filtered.copy()
                    filtered["flow_efficiency"] = filtered["Max Flow (LPM)"] / flow
                    filtered["head_efficiency"] = filtered["Max Head (M)"] / head
                    filtered["total_efficiency"] = filtered["flow_efficiency"] + filtered["head_efficiency"]
                    filtered = filtered.sort_values("total_efficiency")
                    
                    response_text = f"Great! I found {len(filtered)} suitable pump(s) for your requirements ({flow} LPM, {head}m head).\n\n"
                    response_text += "Here are the best matches (sorted by efficiency):\n\n"
                    
                    # Show top 5 results to avoid overwhelming the user
                    top_results = filtered.head(5)
                    for i, (_, row) in enumerate(top_results.iterrows(), 1):
                        response_text += f"**{i}. {row['Model No.']}**\n"
                        response_text += f"   • Max Flow: {row['Max Flow (LPM)']} LPM\n"
                        response_text += f"   • Max Head: {row['Max Head (M)']} m\n"
                        if 'Product Link' in row and pd.notna(row['Product Link']):
                            response_text += f"   • Product Link: {row['Product Link']}\n"
                        response_text += "\n"
                    
                    if len(filtered) > 5:
                        response_text += f"*({len(filtered) - 5} more options available in the detailed results below)*"

            # Simulate typing effect (reduced delay for better UX)
            words = response_text.split()
            displayed_text = ""
            for i, word in enumerate(words):
                displayed_text += word + " "
                if i % 5 == 0:  # Update every 5 words instead of every word
                    message_placeholder.markdown(displayed_text + "▌")
                    time.sleep(0.02)  # Reduced delay
            
            message_placeholder.markdown(response_text)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Display results in a structured table
        if not df.empty and not filtered.empty:
            st.success(f"Found {len(filtered)} matching pump(s).")
            
            # Prepare display columns
            display_columns = ["Model No.", "Max Flow (LPM)", "Max Head (M)"]
            if "Product Link" in filtered.columns:
                display_columns.append("Product Link")
            
            # Show the filtered results
            st.dataframe(
                filtered[display_columns].reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )

            # --- Optional GPT Summary (RAG) ---
            if "OPENAI_API_KEY" in st.secrets:
                with st.expander("💬 AI Recommendation"):
                    try:
                        # Prepare data for AI analysis
                        top_3 = filtered.head(3)
                        pump_data = top_3[["Model No.", "Max Flow (LPM)", "Max Head (M)"]].to_string(index=False)
                        
                        summary_prompt = f"""
You are a pump selection expert. Based on the following pump specifications, recommend the best pump for a user who needs {flow} LPM flow and {head} meters head.

Available pumps (sorted by efficiency):
{pump_data}

Consider:
1. Which pump best matches the requirements without being oversized
2. Energy efficiency (closer to required specs = more efficient)
3. Any other engineering considerations

Provide a clear recommendation with reasoning. Keep it concise and practical.
"""
                        
                        # Use the updated OpenAI API
                        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": summary_prompt}],
                            max_tokens=300,
                            temperature=0.3
                        )
                        
                        st.write(response.choices[0].message.content)
                        
                    except Exception as e:
                        st.error(f"Error generating AI recommendation: {str(e)}")
            else:
                st.info("💡 Add your OpenAI API key to secrets for AI-powered recommendations!")

# --- Sidebar with additional info ---
with st.sidebar:
    st.header("📊 Database Info")
    if not df.empty:
        st.metric("Total Pumps", len(df))
        st.metric("Flow Range", f"{df['Max Flow (LPM)'].min():.0f} - {df['Max Flow (LPM)'].max():.0f} LPM")
        st.metric("Head Range", f"{df['Max Head (M)'].min():.1f} - {df['Max Head (M)'].max():.1f} m")
    else:
        st.warning("Database not accessible")
    
    st.header("💡 Tips")
    st.write("• Choose pumps close to your requirements for better efficiency")
    st.write("• Higher capacity pumps use more energy")
    st.write("• Consider system losses in your calculations")

# --- Clear chat history button ---
if st.button("Clear Chat History"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your pump selection assistant. I can help you find the right pump based on your flow and head requirements. What specifications do you need?"}
    ]
    st.rerun()
