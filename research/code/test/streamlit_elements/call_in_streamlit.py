import streamlit as st
import asyncio
from fastapi import FastAPI
import uvicorn
from threading import Thread

# 1. Setup Streamlit State Tracker
if "api_trigger" not in st.session_state:
    st.session_state.api_trigger = False
if "payload_data" not in st.session_state:
    st.session_state.payload_data = None

# 2. Define the internal function you want to invoke
def internal_target_function(data):
    st.success(f"🎉 Function invoked externally with data: {data}")

# 3. Create the background API server 
app = FastAPI()

@app.post("/trigger")
async def trigger_endpoint(data: dict):
    # Pass data back to Streamlit via state mutation
    st.session_state.api_trigger = True
    st.session_state.payload_data = data
    return {"status": "Success", "message": "Streamlit function queued"}

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

# Start server once on background thread
if "api_started" not in st.session_state:
    Thread(target=run_api, daemon=True).start()
    st.session_state.api_started = True

# 4. Main Streamlit Execution Check
st.title("API-Driven Streamlit App")

if st.session_state.api_trigger:
    # Execute the requested function
    internal_target_function(st.session_state.payload_data)
    
    # Reset states
    st.session_state.api_trigger = False
    st.session_state.payload_data = None

# Fallback UI trigger
if st.button("Manually Run Function"):
    internal_target_function({"source": "UI Button"})
