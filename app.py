import os
import io
import re
import json
import time
import requests
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ==================================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN DARK LUXE (CUSTOM CSS)
# ==================================================
st.set_page_config(
    page_title="GoldLens Feedback Analytics",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Nền ứng dụng Dark Obsidian / Carbon */
    .stApp {
        background-color: #0A0C10;
        color: #E2E8F0;
    }

    /* Thanh Sidebar */
    [data-testid="stSidebar"] {
        background-color: #11141C !important;
        border-right: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    [data-testid="stSidebar"] * {
        color: #C5C9D3 !important;
    }

    /* Header Container */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, rgba(20, 24, 33, 0.95) 0%, rgba(10, 12, 16, 0.98) 100%);
        border: 1px solid rgba(212, 175, 55, 0.25);
        border-radius: 14px;
        padding: 20px 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }

    .brand-title {
        background: linear-gradient(135deg, #BF953F 0%, #FCF6BA 25%, #B38728 50%, #FBF5B7 75%, #AA771C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin: 0;
    }

    .brand-subtitle {
        color: #8A94A6;
        font-size: 13px;
        margin: 3px 0 0 0;
    }

    /* Tiêu đề từng Section */
    .section-title {
        color: #F3E5AB;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        text-transform: uppercase;
    }

    /* Tùy chỉnh Khung Upload File */
    [data-testid="stFileUploader"] {
        background: rgba(20, 24, 33, 0.7);
        border: 2px dashed rgba(212, 175, 55, 0.5) !important;
        border-radius: 14px !important;
        padding: 15px;
        text-align: center;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #D4AF37 !important;
    }

    /* Metrics Tài chính / Phản hồi */
    [data-testid="stMetric"] {
        background: #141821;
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stMetricLabel"] {
        color: #8A94A6 !important;
        font-size: 13px !important;
    }
    [data-testid="stMetricValue"] {
        color: #F5E6AD !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }

    /* Nút bấm Vàng Gold */
    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(135deg, #D4AF37 0%, #AA771C 100%) !important;
        color: #0A0C10 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 12px rgba(212, 175, 55, 0.25) !important;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, #FFE885 0%, #D4AF37 100%) !important;
    }

    /* Style Tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(212, 175, 55, 0.2);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8A94A6;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(212, 175, 55, 0.1) !important;
        color: #F3E5AB !important;
        border-bottom: 2px solid #D4AF37 !important;
    }

    header[data-testid="stHeader"] { background: transparent; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# 2. KHỞI TẠO SESSION STATE & CẤU HÌNH GEMINI MODEL
# ==================================================
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if "voc_df" not in st.session_state:
    st.session_state.voc_df = None

if "voc_errors" not in st.session_state:
    st.session_state.voc_errors = pd.DataFrame()

if "voc_summary" not in st.session_state:
    st.session_state.voc_summary = ""

def get_gemini_client(input_key=""):
    key = input_key.strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None

# ==================================================
# 3. SIDEBAR NAVIGATION & BỘ LỌC
# ==================================================
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; padding: 10px 0 20px 0; border-bottom: 1px solid rgba(212,175,55,0.15);">
            <svg width="36" height="36" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="50" cy="50" r="45" stroke="url(#gold-grad)" stroke-width="6"/>
                <path d="M50 20 L63 40 L85 50 L63 60 L50 80 L37 60 L15 50 L37 40 Z" fill="url(#gold-grad)"/>
                <defs>
                    <linearGradient id="gold-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#FCF6BA" />
                        <stop offset="50%" stop-color="#D4AF37" />
                        <stop offset="100%" stop-color="#AA771C" />
                    </linearGradient>
                </defs>
            </svg>
            <div>
                <div style="font-size: 16px; font-weight: 800; color: #F3E5AB;">GOLDLENS</div>
                <div style="font-size: 10px; color: #788294; letter-spacing: 1px;">FEEDBACK ANALYTICS</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    api_key_input = st.text_input("🔑 Gemini API Key", type="password", help="Nhập Gemini API Key của bạn nếu chưa cài biến môi trường")
    client = get_gemini_client(api_key_input)

    page = st.radio(
        label="MENU ĐIỀU HƯỚNG",
        options=[
            "Trang chủ",
            "Phân tích khách hàng",
            "Dashboard",
            "Báo cáo quản trị AI",
            "Tin tức Vietstock"
        ]
    )

    st.markdown("---")
    st.caption("BỘ LỌC DỮ LIỆU")
    st.selectbox("Nguồn phản hồi", ["Tất cả", "Google Forms", "Zendesk", "File Upload"])
    st.selectbox("Khoảng thời gian", ["12 tháng qua", "6 tháng qua", "30 ngày qua"])

    st.markdown("---")
    if client:
        st.success(f"🟢 Gemini API đã sẵn sàng ({MODEL_NAME})")
    else:
        st.warning("🟡 Chưa kết nối Gemini API Key")

# ==================================================
# 4. HEADER BAR
# ==================================================
st.markdown("""
<div class="header-container">
    <div style="display: flex; align-items: center; gap: 16px;">
        <svg width="48" height="48" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="44" stroke="url(#header-gold)" stroke-width="5"/>
            <polygon points="50,15 62,38 85,50 62,62 50,85 38,62 15,50 38,38" fill="url(#header-gold)"/>
            <circle cx="50" cy="50" r="8" fill="#0A0C10"/>
            <defs>
                <linearGradient id="header-gold" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#FFE99B" />
                    <stop offset="50%" stop-color="#D4AF37" />
                    <stop offset="100%" stop-color="#AA771C" />
                </linearGradient>
            </defs>
        </svg>
        <div>
            <h1 class="brand-title">GOLDLENS FEEDBACK ANALYTICS</h1>
            <p class="brand-subtitle">Thấu hiểu phản hồi, nâng tầm dịch vụ</p>
        </div>
    </div>
