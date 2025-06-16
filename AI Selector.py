# app.py
import streamlit as st
import pandas as pd
import openai
from supabase import create_client
import time

# --- Initialize Supabase ---
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase = create_client(url, key)

# --- Load pump data from Supabase ---
@st.cache_data
def load_data():
    response = supabase.table("pump_specs").select("*").execute()
    return pd.DataFrame(response.data)

df = load_data()

# --- App UI ---
st.title("🔍 Pump Selector Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your pump selection assistant. I can help you find the right pump based on your flow and head requirements. What specifications do you need?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input ---
flow = st.number_input("Required Flow (LPM)", min_value=1)
head = st.number_input("Required Head (m)", min_value=1)

# --- Search and Filter ---
if st.button("Search"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": f"I need a pump with {flow} LPM flow and {head} meters head."})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(f"I need a pump with {flow} LPM flow and {head} meters head.")

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Ensure numeric values for filtering
        df["Max Flow (LPM)"] = pd.to_numeric(df["Max Flow (LPM)"], errors="coerce")
        df["Max Head (M)"] = pd.to_numeric(df["Max Head (M)"], errors="coerce")

        filtered = df[
            (df["Max Flow (LPM)"] >= flow) &
            (df["Max Head (M)"] >= head)
        ]

        # Prepare response
        if filtered.empty:
            response_text = "I couldn't find any suitable pumps for your requirements. Would you like to try different specifications?"
        else:
            response_text = f"I found {len(filtered)} matching pump(s) for your requirements. Here are the details:\n\n"
            for _, row in filtered.iterrows():
                response_text += f"Model: {row['Model No.']}\n"
                response_text += f"Max Flow: {row['Max Flow (LPM)']} LPM\n"
                response_text += f"Max Head: {row['Max Head (M)']} m\n"
                response_text += f"Product Link: {row['Product Link']}\n\n"

        # Simulate typing effect
        for chunk in response_text.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Display results in a more structured way
    if not filtered.empty:
        st.success(f"Found {len(filtered)} matching pump(s).")
        st.dataframe(filtered[["Model No.", "Max Flow (LPM)", "Max Head (M)", "Product Link"]])

        # --- Optional GPT Summary (RAG) ---
        with st.expander("💬 AI Suggestion"):
            summary_prompt = f"""
You are a pump selection assistant. Based only on the following pump data, suggest a suitable model for a user needing {flow} LPM and {head} meters head.

Data:
{filtered[["Model No.", "Max Flow (LPM)", "Max Head (M)"]].to_string(index=False)}

Do not make assumptions beyond this data. If uncertain, say no match found.
"""
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": summary_prompt}]
            )
            st.write(response.choices[0].message["content"])
