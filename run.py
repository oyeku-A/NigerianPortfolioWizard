import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import streamlit as st

# Configure page - MUST be the first Streamlit command
st.set_page_config(
    page_title="Nigerian Portfolio Wizard",
    page_icon="📈",
    layout="wide"
)

# Import and run the main app
from app.app import main

if __name__ == "__main__":
    main() 