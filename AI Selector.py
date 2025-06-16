# app.py
import streamlit as st
import pandas as pd
import openai
from supabase import create_client

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
st.title("🔍 Pump Selector Assistant (RAG-based, Supabase)")

# --- User Input ---
flow = st.number_input("Required Flow (LPM)", min_value=1)
head = st.number_input("Required Head (m)", min_value=1)

# --- Search and Filter ---
if st.button("Search"):
    # Ensure numeric values for filtering
    df["Max Flow (LPM)"] = pd.to_numeric(df["Max Flow (LPM)"], errors="coerce")
    df["Max Head (M)"] = pd.to_numeric(df["Max Head (M)"], errors="coerce")

    filtered = df[
        (df["Max Flow (LPM)"] >= flow) &
        (df["Max Head (M)"] >= head)
    ]

    # Display results
    if filtered.empty:
        st.warning("No suitable pump found for your requirements.")
    else:
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
