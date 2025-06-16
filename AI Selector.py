import streamlit as st
import pandas as pd
from supabase import create_client
import openai

# --- Init Supabase ---
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase = create_client(url, key)

# --- Init OpenAI ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Load Data from Supabase ---
@st.cache_data
def load_data():
    response = supabase.table("pump_specs").select("*").execute()
    return pd.DataFrame(response.data)

df = load_data()
df["Max Flow (LPM)"] = pd.to_numeric(df["Max Flow (LPM)"], errors="coerce")
df["Max Head (M)"] = pd.to_numeric(df["Max Head (M)"], errors="coerce")

# --- Chat UI ---
st.title("💬 Pump Selector Chat (GPT + Supabase)")

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! 👋 Tell me your required flow and head, and I’ll help you choose a pump."}]

# Show past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if user_input := st.chat_input("e.g. I need a pump for 50 LPM at 10 meters"):
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Very basic NLP: extract numbers
    import re
    flow_matches = re.findall(r"(\\d+\\.?\\d*)\\s*(?:LPM|liters per minute)", user_input, re.IGNORECASE)
    head_matches = re.findall(r"(\\d+\\.?\\d*)\\s*(?:m|meters|metres)", user_input, re.IGNORECASE)

    try:
        flow = float(flow_matches[0]) if flow_matches else None
        head = float(head_matches[0]) if head_matches else None
    except:
        flow, head = None, None

    # Assistant reply box
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if not flow or not head:
            full_response = "❌ I couldn't detect both flow and head in your message. Please try again with something like: '50 LPM at 10 meters'."
        else:
            # Filter pump data
            filtered = df[
                (df["Max Flow (LPM)"] >= flow) &
                (df["Max Head (M)"] >= head)
            ]

            if filtered.empty:
                full_response = f"😕 I couldn’t find any pumps matching {flow} LPM at {head} meters head."
            else:
                pump_summary = filtered[["Model No.", "Max Flow (LPM)", "Max Head (M)"]].to_string(index=False)
                prompt = f"""
You are a helpful pump selection assistant. Based ONLY on the following data, suggest a suitable model for a user needing {flow} LPM and {head} meters head.

Data:
{pump_summary}

DO NOT guess or invent models. If no suitable pump exists, say so clearly.
"""
                gpt_response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                full_response = gpt_response.choices[0].message["content"]

        # Typing animation
        display_text = ""
        for chunk in full_response.split():
            display_text += chunk + " "
            message_placeholder.markdown(display_text + "▌")
            import time
            time.sleep(0.03)
        message_placeholder.markdown(display_text)

        st.session_state.messages.append({"role": "assistant", "content": display_text})
