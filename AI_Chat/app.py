import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import json
import os
import sys
from textblob import TextBlob
import logging
from transformers import pipeline
import torch
import numpy as np
from fuzzywuzzy import process
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import re
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib
from GoogleNews import GoogleNews
matplotlib.use('Agg')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from AI_Chat.chatbot import IndianStockChatbot as MLChatbot
from Portfolio.portfolio import load_portfolio, calculate_portfolio_metrics
from Watchlist.watchlist_operations import WatchlistManager

            "sbi": "SBIN",

def show_chat():
    """Display the AI chat page"""
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = [
            "What's the price of TCS?",
            "Show market activity",
            "What's the sentiment for Reliance?",
            "Show my watchlist",
            "Show my portfolio"
        ]

    # Custom CSS
    st.markdown("""
    <style>
        body, .main, .block-container {
            background-color: #101820 !important;
        }
        .stApp {
            background-color: #101820 !important;
        }
        .stTextInput>div>div>input {
            font-size: 1.1rem;
            background-color: #1B2C3A !important;
            color: #fff !important;
            border: 1.5px solid #223344 !important;
        }
        .chat-outer {
            max-height: 480px;
            min-height: 320px;
            overflow-y: auto;
            padding-bottom: 1rem;
            margin-bottom: 1.2rem;
            border-radius: 1.2rem;
            background: rgba(27,44,58,0.13);
            box-shadow: 0 2px 12px rgba(27,44,58,0.10);
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 2rem;
            margin-bottom: 0.5rem;
        }
        .chat-message {
            display: flex;
            align-items: flex-end;
            gap: 1.2rem;
            max-width: 80%;
            border-radius: 1.7rem;
            box-shadow: 0 4px 24px rgba(27,44,58,0.13);
            padding: 1.5rem 1.7rem;
            margin-bottom: 0.7rem;
            font-size: 1.13rem;
            word-break: break-word;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            opacity: 1;
            transition: box-shadow 0.2s, opacity 0.4s;
        }
        .chat-message.bot.new {
            animation: fadeInBot 0.7s;
        }
        @keyframes fadeInBot {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .chat-message.user {
            background: linear-gradient(135deg, #1B2C3A 85%, #223344 100%);
            color: #fff;
            margin-left: auto;
            border-bottom-right-radius: 0.7rem;
            border-top-right-radius: 0.7rem;
            border-bottom-left-radius: 1.7rem;
            border-top-left-radius: 1.7rem;
            border: 2px solid #223344;
        }
        .chat-message.bot {
            background: linear-gradient(135deg, #223344 85%, #1B2C3A 100%);
            color: #e6eaf0;
            margin-right: auto;
            border-bottom-left-radius: 0.7rem;
            border-top-left-radius: 0.7rem;
            border-bottom-right-radius: 1.7rem;
            border-top-right-radius: 1.7rem;
            border: 2px solid #1B2C3A;
        }
        .chat-avatar {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.7rem;
            font-weight: bold;
            box-shadow: 0 2px 8px rgba(27,44,58,0.13);
        }
        .chat-avatar.user {
            background: #1B2C3A;
            color: #fff;
            border: 2.5px solid #223344;
        }
        .chat-avatar.bot {
            background: #223344;
            color: #fff;
            border: 2.5px solid #1B2C3A;
        }
        .chat-message .content {
            margin-top: 0;
            flex: 1;
        }
        .sticky-input {
            position: sticky;
            bottom: 0;
            background: #101820;
            z-index: 10;
            padding-top: 1rem;
            padding-bottom: 0.5rem;
        }
        .input-separator {
            border: none;
            border-top: 2px solid #223344;
            margin: 0.5rem 0 1.2rem 0;
        }
        .loading-bot {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin-left: 2.5rem;
            margin-bottom: 1.2rem;
        }
        .loading-dots span {
            display: inline-block;
            width: 10px;
            height: 10px;
            margin-right: 2px;
            background: #e6eaf0;
            border-radius: 50%;
            opacity: 0.7;
            animation: loadingFade 1.2s infinite;
        }
        .loading-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }
        .loading-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }
        @keyframes loadingFade {
            0%, 80%, 100% { opacity: 0.3; }
            40% { opacity: 1; }
        }
        .stButton>button {
            width: 100%;
            border-radius: 0.7rem;
            height: 3.2rem;
            font-size: 1.13rem;
            background-color: #1B2C3A !important;
            color: #fff !important;
            border: 1.5px solid #223344 !important;
            font-weight: 600;
            transition: background 0.2s, color 0.2s;
        }
        .stButton>button:hover {
            background-color: #223344 !important;
            color: #fff !important;
        }
        .suggestion-chip {
            display: inline-block;
            padding: 0.6rem 1.2rem;
            margin: 0.3rem;
            border-radius: 1.2rem;
            background-color: #1B2C3A;
            color: #fff;
            cursor: pointer;
            border: 1.5px solid #223344;
            font-size: 1.05rem;
            font-weight: 500;
            box-shadow: 0 2px 8px rgba(27,44,58,0.10);
        }
        .suggestion-chip:hover {
            background-color: #223344;
            color: #fff;
        }
        .stMetric {
            background-color: #1B2C3A !important;
            color: #fff !important;
            border-radius: 0.7rem;
            padding: 0.7rem 1.2rem;
            box-shadow: 0 2px 8px rgba(27,44,58,0.10);
        }
        .stExpander, .stExpanderHeader {
            background-color: #1B2C3A !important;
            color: #fff !important;
            border-radius: 0.7rem !important;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #fff !important;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .section-card {
            background: linear-gradient(135deg, #1B2C3A 90%, #223344 100%);
            border-radius: 1.2rem;
            box-shadow: 0 4px 24px rgba(27,44,58,0.13);
            padding: 1.5rem 1.7rem;
            margin-bottom: 1.5rem;
            color: #fff;
        }
        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 1rem;
            letter-spacing: 0.5px;
        }
        .section-divider {
            border: none;
            border-top: 2px solid #223344;
            margin: 1.2rem 0 1.5rem 0;
        }
        .stInfo, .stAlert, .stError {
            background-color: #223344 !important;
            color: #fff !important;
            border-radius: 0.7rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize chatbot with loading state
    if not st.session_state.initialized:
        with st.spinner("Initializing chatbot and loading models... This may take a few minutes."):
            try:
                st.session_state.chatbot = MLChatbot()
                st.session_state.initialized = True
                st.success("Chatbot initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing chatbot: {str(e)}")
                st.info("The chatbot will continue with limited functionality.")

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💬 AI Chat Assistant</div>', unsafe_allow_html=True)
        st.write("Ask about Indian stocks, indices, market activity, portfolio, sentiment, or watchlist.")

        # Chat area with fixed height and scroll
        st.markdown('<div class="chat-outer">', unsafe_allow_html=True)
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for idx, (user_msg, bot_msg) in enumerate(st.session_state.history):
            st.markdown(f'''
            <div style="display: flex; flex-direction: row-reverse; align-items: flex-end;">
                <div class="chat-avatar user">🧑</div>
                <div class="chat-message user">
                    <div class="content">{user_msg}</div>
                </div>
            </div>
            <div style="display: flex; align-items: flex-end;">
                <div class="chat-avatar bot">🤖</div>
                <div class="chat-message bot{' new' if idx == len(st.session_state.history)-1 else ''}">
                    <div class="content">{bot_msg}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        # Loading animation if bot is processing
        if st.session_state.get('bot_loading', False):
            st.markdown('''
            <div class="loading-bot">
                <div class="chat-avatar bot">🤖</div>
                <div class="loading-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Quick suggestions
        st.markdown('<div style="margin-bottom: 1.2rem;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size:1.08rem; margin-bottom:0.5rem;">💡 Quick Suggestions</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, suggestion in enumerate(st.session_state.suggestions):
            if cols[i % 4].button(suggestion, key=f"suggestion_{i}", help="Click to use this suggestion"):
                st.session_state.user_input = suggestion
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Visually separate input area
        st.markdown('<hr class="input-separator" />', unsafe_allow_html=True)
        st.markdown('<div class="sticky-input">', unsafe_allow_html=True)
        # User input
        user_input = st.text_input("Type your question:", key="user_input")
        # Send button
        send_clicked = False
        if 'user_input' in st.session_state and st.session_state.user_input:
             send_clicked = st.button("Send", key="send_button") or (st.session_state.user_input and st.session_state.get('last_input') != st.session_state.user_input)
        else:
             send_clicked = st.button("Send", key="send_button")
        if send_clicked:
            if user_input.strip():
                st.session_state['bot_loading'] = True
                with st.spinner("Processing your query..."):
                    try:
                        bot_response = st.session_state.chatbot.process_query(user_input)
                        st.session_state.history.append((user_input, bot_response))
                        st.session_state.last_input = user_input
                        st.session_state['bot_loading'] = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        st.session_state.history.append((user_input, "I'm having trouble processing your request. Please try again."))
                        st.session_state['bot_loading'] = False
        st.markdown('</div>', unsafe_allow_html=True)
        # Clear chat button
        if st.button("Clear Chat", key="clear_button"):
            st.session_state.history = []
            st.session_state.last_input = ""
            st.rerun()

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Market Overview</div>', unsafe_allow_html=True)
        try:
            market_data = st.session_state.chatbot.get_market_activity()
            if market_data:
                # Nifty 50
                st.metric(
                    "Nifty 50",
                    f"₹{market_data['nifty']['current']:,.2f}",
                    f"{market_data['nifty']['change_pct']:+.2f}%"
                )
                
                # Sensex
                st.metric(
                    "Sensex",
                    f"₹{market_data['sensex']['current']:,.2f}",
                    f"{market_data['sensex']['change_pct']:+.2f}%"
                )
                
                # Market Status
                status_color = "#00e676" if market_data['market_status'] == "Open" else "#ff5252"
                st.markdown(f"Market Status: <span style='color:{status_color}; font-weight:600;'>{market_data['market_status']}</span>", unsafe_allow_html=True)
                
                # Advance-Decline Ratio
                if market_data['advance_decline']['ratio'] != float('inf'):
                    st.metric(
                        "Advance-Decline Ratio",
                        f"{market_data['advance_decline']['ratio']:.2f}",
                        f"Advances: {market_data['advance_decline']['advances']} | Declines: {market_data['advance_decline']['declines']}"
                    )
        except Exception as e:
            st.error("Unable to fetch market data")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔍 Recent Searches</div>', unsafe_allow_html=True)
        if st.session_state.history:
            recent_searches = [msg[0] for msg in st.session_state.history[-5:]]
            for i, search in enumerate(recent_searches):
                if st.button(search, key=f"recent_search_{i}_{hash(search)}"):
                    st.session_state.user_input = search
                    st.rerun()
        else:
            st.info("No recent searches")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💡 Help & Tips</div>', unsafe_allow_html=True)
        with st.expander("Show Help & Tips"):
            st.markdown("""
            **Example Queries:**
            - Stock Price: "What's the price of TCS?"
            - Market Activity: "Show market activity"
            - Sentiment: "What's the sentiment for Reliance?"
            - Watchlist: "Show my watchlist"
            - Market Terms: "Explain what is IPO?"
            - Sector Analysis: "Show sector performance"
            - Trading Signals: "What are the trading signals for HDFC Bank?"
            - Portfolio: "Show my portfolio analysis"
            
            **Tips:**
            - Be specific with company names
            - Use stock symbols for faster results
            - Ask about market terms for explanations
            - Use the quick suggestions for common queries
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Powered by AI • Real-time Market Data • Last Updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    show_chat() 
