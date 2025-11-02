import streamlit as st
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from Dashboard.streamlit_dashboard import show_dashboard
from Market_Activity.market_activity import show_market_activity
from Watchlist.watchlist_page import show_watchlist
from Portfolio.portfolio import show_portfolio
from AI_Analysis.ai_analysis import show_ai_analysis
from AI_Chat.app import show_chat

# Page configuration
st.set_page_config(
    page_title="StockRaj - Indian Stock Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1e2130;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
    }
    
    /* Cards */
    .stCard {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2b313e;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #475063;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #1e2130;
        border-radius: 5px;
        padding: 15px;
    }
    
    /* Charts */
    .js-plotly-plot {
        background-color: #1e2130;
        border-radius: 10px;
    }
    
    /* Tables */
    .stDataFrame {
        background-color: #1e2130;
        border-radius: 10px;
    }
    
    /* Navigation */
    .nav-link {
        color: #ffffff;
        text-decoration: none;
        padding: 10px 20px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .nav-link:hover {
        background-color: #2b313e;
    }
    
    .nav-link.active {
        background-color: #475063;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #666666;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for current page
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Dashboard'

# Sidebar navigation
with st.sidebar:
    st.title("📈 StockRaj")
    st.markdown("---")
    
    # Navigation buttons
    pages = {
        'Dashboard': '📊',
        'Market Activity': '📈',
        'Watchlist': '👀',
        'Portfolio': '💼',
        'AI Analysis': '🤖',
        'AI Chat': '💬'
    }
    
    for page, icon in pages.items():
        if st.button(f"{icon} {page}", key=page, use_container_width=True):
            st.session_state.current_page = page
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Powered by AI • Real-time Data</p>
    </div>
    """, unsafe_allow_html=True)

# Main content area
if st.session_state.current_page == 'Dashboard':
    show_dashboard()
elif st.session_state.current_page == 'Market Activity':
    show_market_activity()
elif st.session_state.current_page == 'Watchlist':
    show_watchlist()
elif st.session_state.current_page == 'Portfolio':
    show_portfolio()
elif st.session_state.current_page == 'AI Analysis':
    show_ai_analysis()
elif st.session_state.current_page == 'AI Chat':
    show_chat()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>© 2024 StockRaj • All rights reserved</p>
</div>
""", unsafe_allow_html=True) 