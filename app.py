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
# 2. KHỞI TẠO SESSION STATE & CẤU HÌNH GEMINI
# ==================================================
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
        st.success("🟢 Gemini API đã sẵn sàng")
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
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="color: #4ADE80; font-size: 13px;">● Live Analytics</span>
        <div style="background: rgba(212,175,55,0.15); border: 1px solid rgba(212,175,55,0.3); padding: 6px 14px; border-radius: 20px; font-size: 13px; color: #F3E5AB;">
            👤 Trrdng hoc
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# 5. HÀM XỬ LÝ DỮ LIỆU & AI (CÓ RETRY LỖI 429 RATE LIMIT)
# ==================================================
def find_column(df, candidates):
    columns = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")
        if key in columns:
            return columns[key]
    return None

def load_feedback(uploaded_file):
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = [str(c).strip() for c in df.columns]

    text_col = find_column(df, ["Feedback_Text", "feedback", "comment", "review", "content", "text", "customer_feedback", "noi_dung"])
    if text_col is None:
        text_cols = list(df.select_dtypes(include=["object", "string"]).columns)
        if not text_cols:
            raise ValueError("Không tìm thấy cột nội dung feedback.")
        text_col = max(text_cols, key=lambda c: df[c].fillna("").astype(str).str.len().mean())

    if text_col != "Feedback_Text":
        df["Feedback_Text"] = df[text_col]

    df["Feedback_Text"] = df["Feedback_Text"].fillna("").astype(str).str.strip()
    df = df[df["Feedback_Text"] != ""].copy()

    if df.empty:
        raise ValueError("File không có feedback hợp lệ.")

    if "Feedback_ID" not in df.columns:
        df.insert(0, "Feedback_ID", [f"FB{i:05d}" for i in range(1, len(df) + 1)])

    date_col = find_column(df, ["Feedback_Date", "date", "created_at", "review_date", "ngay", "ngày"])
    if date_col and date_col != "Feedback_Date":
        df["Feedback_Date"] = df[date_col]
    elif "Feedback_Date" not in df.columns:
        df["Feedback_Date"] = pd.NaT

    df["Feedback_Date"] = pd.to_datetime(df["Feedback_Date"], errors="coerce")
    return df.reset_index(drop=True)

def classify_data_batch(df, client_obj, batch_size=15):
    """Phân loại phản hồi theo Batch có bọc Retry xử lý triệt để lỗi 429 Rate Limit"""
    out_df = df.copy()
    out_df["Product"] = ""
    out_df["Topic"] = ""
    out_df["Sentiment"] = ""
    out_df["Urgency"] = ""
    
    errors = []
    total_rows = len(out_df)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = out_df.iloc[start_idx:end_idx]

        batch_data = [
            {"id": int(i), "text": str(row["Feedback_Text"])}
            for i, row in batch_df.iterrows()
        ]

        prompt = f"""
        Bạn là chuyên gia phân tích Voice of Customer (VOC) cho ngân hàng.
        Phân loại danh sách feedback sau:

        Quy tắc giá trị:
        - Product: Mobile Banking, Transfer, ATM, Credit Card, Debit Card, Loan, Savings, Branch Service, Other
        - Topic: Login, Transaction, App Performance, Fee, Customer Service, Security, ATM Failure, Card, Loan, Interest Rate, Other
        - Sentiment: Positive, Neutral, Negative
        - Urgency: Low, Medium, High, Critical

        Danh sách Feedback:
        {json.dumps(batch_data, ensure_ascii=False)}

        Trả về duy nhất mảng JSON chứa các object có cấu trúc:
        [
          {{"id": 0, "Product": "...", "Topic": "...", "Sentiment": "...", "Urgency": "..."}},
          ...
        ]
        """

        # Vòng lặp Retry xử lý lỗi 429
        while True:
            try:
                response = client_obj.models.generate_content(
                    model="gemini-3.8-flash ",
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                results = json.loads(response.text)
                for item in results:
                    idx = item.get("id")
                    if idx in out_df.index:
                        out_df.at[idx, "Product"] = item.get("Product", "Other")
                        out_df.at[idx, "Topic"] = item.get("Topic", "Other")
                        out_df.at[idx, "Sentiment"] = item.get("Sentiment", "Neutral")
                        out_df.at[idx, "Urgency"] = item.get("Urgency", "Low")
                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    status_text.warning(f"⏳ Bị nghẽn Quota (429) ở dòng {start_idx+1}-{end_idx}. Đang chờ 35 giây...")
                    time.sleep(35)
                else:
                    status_text.error(f"❌ Lỗi dòng {start_idx+1}-{end_idx}: {e}")
                    for idx in batch_df.index:
                        errors.append({"index": idx, "text": batch_df.at[idx, "Feedback_Text"], "error": str(e)})
                    break

        time.sleep(3)  # Giữ nhịp để tránh quá tải
        progress_bar.progress(end_idx / total_rows)
        status_text.text(f"Đã xử lý {end_idx}/{total_rows} dòng...")

    progress_bar.empty()
    status_text.empty()
    return out_df, pd.DataFrame(errors)

def make_clusters(df, number_of_clusters=5):
    out = df.copy()
    texts = out["Feedback_Text"].fillna("").astype(str)

    if len(out) < 2:
        out["Cluster"] = 0
        out["Natural_Topic"] = "Nhóm feedback"
        return out

    k = min(max(2, int(number_of_clusters)), len(out))

    try:
        vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts)
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(matrix)

        out["Cluster"] = labels
        terms = np.array(vectorizer.get_feature_names_out())
        names = {}

        for cluster in sorted(set(labels)):
            mask = labels == cluster
            mean_vector = np.asarray(matrix[mask].mean(axis=0)).ravel()
            top_indices = mean_vector.argsort()[-3:][::-1]
            keywords = ", ".join(terms[top_indices])

            mode = out.loc[mask, "Topic"].mode()
            topic = mode.iloc[0] if not mode.empty else "Feedback"
            names[cluster] = f"{topic} — {keywords}"

        out["Natural_Topic"] = out["Cluster"].map(names)
    except Exception:
        out["Cluster"] = 0
        out["Natural_Topic"] = "Nhóm feedback"

    return out

def export_excel(df, errors):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="VoC_Results")
        if not errors.empty:
            errors.to_excel(writer, index=False, sheet_name="Errors")
    return buffer.getvalue()

@st.cache_data(ttl=900, show_spinner=False)
def get_vietstock_news():
    feeds = [
        "https://vietstock.vn/732/chung-khoan.rss",
        "https://vietstock.vn/733/doanh-nghiep.rss",
        "https://vietstock.vn/734/tai-chinh.rss",
        "https://vietstock.vn/761/kinh-te/vi-mo.rss"
    ]
    items = []
    for url in feeds:
        try:
            response = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            root = ET.fromstring(response.content)

            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                description = (item.findtext("description") or "")
                published = (item.findtext("pubDate") or "").strip()

                description = re.sub(r"<[^>]+>", " ", description)
                description = re.sub(r"\s+", " ", description).strip()

                if title and link:
                    items.append({"title": title, "link": link, "description": description, "published": published})
        except Exception:
            continue

    unique = {item["link"]: item for item in items}
    return list(unique.values())

# ==================================================
# 6. NỘI DUNG CÁC TRANG (PAGES)
# ==================================================

# --------------------------------------------------
# TRANG 1: TRANG CHỦ
# --------------------------------------------------
if page == "Trang chủ":
    st.markdown('<p class="section-title">SECTION 1: TỔNG QUAN HỆ THỐNG GOLDLENS</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="💬 Tổng phản hồi hệ thống", value="15,240", delta="+5% tháng này")
    with col2:
        st.metric(label="🙂 Cảm xúc Tích cực", value="68%", delta="+68% tháng này")
    with col3:
        st.metric(label="☹️ Cảm xúc Tiêu cực", value="18%", delta="-18% tháng này")
    with col4:
        st.metric(label="👥 Vấn đề chính", value="Chờ đợi, Thẻ, App")

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Xu hướng cảm xúc khách hàng theo thời gian")
    months = [f"{i} tháng" for i in range(1, 13)]
    positive_trend = [48, 70, 58, 75, 52, 58, 88, 62, 92, 48, 108, 52]
    negative_trend = [42, 20, 52, 30, 42, 32, 22, 38, 82, 28, 58, 48]
    neutral_trend  = [42, 48, 42, 52, 48, 58, 38, 65, 68, 55, 52, 45]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=positive_trend, mode='lines+markers', name='Positive', line=dict(color='#22C55E', width=2), fill='tonexty', fillcolor='rgba(34, 197, 94, 0.08)'))
    fig.add_trace(go.Scatter(x=months, y=negative_trend, mode='lines+markers', name='Negative', line=dict(color='#EF4444', width=2), fill='tonexty', fillcolor='rgba(239, 68, 68, 0.08)'))
    fig.add_trace(go.Scatter(x=months, y=neutral_trend, mode='lines+markers', name='Neutral', line=dict(color='#EAB308', width=2), fill='tonexty', fillcolor='rgba(234, 179, 8, 0.08)'))

    fig.update_layout(
        paper_bgcolor='#141821',
        plot_bgcolor='#141821',
        margin=dict(l=20, r=20, t=30, b=30),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#8A94A6'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#8A94A6'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#C5C9D3')),
        height=360
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🧠 Customer Intelligence")
        st.write("Phân loại tự động Sản phẩm, Chủ đề, Cảm xúc & Độ khẩn cấp bằng Gemini Flash.")
    with c2:
        st.markdown("### 📊 Smart Dashboard")
        st.write("Biểu đồ tương tác thời gian thực, phát hiện cảnh báo sớm rủi ro dịch vụ.")
    with c3:
        st.markdown("### 📰 Vietstock News")
        st.write("Cập nhật dòng chảy tin tức tài chính, kinh tế và thị trường liên tục.")

# --------------------------------------------------
# TRANG 2: PHÂN TÍCH KHÁCH HÀNG
# --------------------------------------------------
elif page == "Phân tích khách hàng":
    st.title("🗄️ Phân tích Voice of Customer")

    uploaded = st.file_uploader(
        "Kéo thả tệp dữ liệu phản hồi (.csv, .xlsx) của bạn tại đây",
        type=["xlsx", "xls", "csv"]
    )

    number_of_clusters = st.slider("Số nhóm Natural Topic (Clustering):", min_value=2, max_value=10, value=5)

    if uploaded is not None:
        try:
            preview = load_feedback(uploaded)
            st.success(f"Đã đọc thành công {len(preview)} dòng dữ liệu.")
            st.dataframe(preview.head(5), use_container_width=True)

            if st.button("🚀 Chạy phân tích AI & Gom cụm", type="primary"):
                if client is None:
                    st.error("⚠️ Vui lòng nhập Gemini API Key ở thanh bên (Sidebar) để chạy!")
                else:
                    with st.spinner("Đang phân loại phản hồi bằng Gemini AI (Batch Mode)..."):
                        result, errors = classify_data_batch(preview, client)

                    with st.spinner("Đang gom cụm chủ đề tự nhiên (Clustering)..."):
                        result = make_clusters(result, number_of_clusters)

                    if "Feedback_Date" in result.columns and not result["Feedback_Date"].isna().all():
                        result["Month"] = result["Feedback_Date"].dt.to_period("M").astype(str)

                    st.session_state.voc_df = result
                    st.session_state.voc_errors = errors
                    st.session_state.voc_summary = ""
                    st.success("🎉 Đã hoàn tất phân tích toàn bộ dữ liệu!")

        except Exception as error:
            st.error(f"❌ Không thể đọc hoặc xử lý file: {error}")

    df_res = st.session_state.voc_df

    if df_res is not None:
        st.markdown("---")
        st.subheader("📋 Kết quả phân loại chi tiết")

        search_query = st.text_input("🔍 Tìm kiếm trong kết quả:")
        display_df = df_res.copy()

        if search_query:
            mask = display_df.astype(str).apply(lambda col: col.str.contains(search_query, case=False, na=False)).any(axis=1)
            display_df = display_df[mask]

        st.dataframe(display_df, use_container_width=True, height=400)

        st.download_button(
            "⬇️ Tải kết quả Excel",
            data=export_excel(df_res, st.session_state.voc_errors),
            file_name="GoldLens_VoC_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# --------------------------------------------------
# TRANG 3: DASHBOARD
# --------------------------------------------------
elif page == "Dashboard":
    st.title("📊 Customer Intelligence Dashboard")

    df_res = st.session_state.voc_df

    if df_res is None:
        st.info("💡 Chưa có dữ liệu phân tích. Vui lòng mở mục **Phân tích khách hàng** để tải file và chạy AI trước.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng feedback", f"{len(df_res):,}")
        c2.metric("Tiêu cực", int((df_res["Sentiment"] == "Negative").sum()))
        c3.metric("Khẩn cấp / Cao", int(df_res["Urgency"].isin(["High", "Critical"]).sum()))
        c4.metric("Nhóm chủ đề", int(df_res["Natural_Topic"].nunique()))

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            sentiment_df = df_res["Sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Count")
            fig_pie = px.pie(
                sentiment_df, names="Sentiment", values="Count", title="Phân bổ cảm xúc", hole=0.4,
                color="Sentiment", color_discrete_map={"Positive": "#22C55E", "Neutral": "#EAB308", "Negative": "#EF4444"}
            )
            fig_pie.update_layout(paper_bgcolor='#141821', plot_bgcolor='#141821', font=dict(color='#E2E8F0'))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            topic_df = df_res["Topic"].value_counts().rename_axis("Topic").reset_index(name="Count")
            fig_bar = px.bar(topic_df, x="Topic", y="Count", title="Số lượng Feedback theo Chủ đề", color_discrete_sequence=['#D4AF37'])
            fig_bar.update_layout(paper_bgcolor='#141821', plot_bgcolor='#141821', font=dict(color='#E2E8F0'))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("🚨 Cảnh báo khẩn cấp & Tiêu cực")
        urgent_df = df_res[df_res["Urgency"].isin(["High", "Critical"]) | (df_res["Sentiment"] == "Negative")]

        if urgent_df.empty:
            st.success("Không phát hiện feedback tiêu cực hoặc khẩn cấp.")
        else:
            st.warning(f"Phát hiện {len(urgent_df)} feedback cần rà soát xử lý ngay:")
            st.dataframe(
                urgent_df[["Feedback_ID", "Feedback_Text", "Product", "Topic", "Sentiment", "Urgency"]],
                use_container_width=True
            )

# --------------------------------------------------
# TRANG 4: BÁO CÁO QUẢN TRỊ AI
# --------------------------------------------------
elif page == "Báo cáo quản trị AI":
    st.title("📄 AI Management Summary")

    df_res = st.session_state.voc_df

    if df_res is None:
        st.info("💡 Vui lòng phân tích dữ liệu ở trang **Phân tích khách hàng** trước khi tạo báo cáo.")
    elif client is None:
        st.warning("⚠️ Vui lòng cấu hình Gemini API Key ở Sidebar để sử dụng tính năng này.")
    else:
        st.write("Tạo báo cáo tổng hợp dành cho Ban Quản trị bằng mô hình Gemini AI.")

        if st.button("✨ Sinh báo cáo quản trị AI", type="primary"):
            stats = {
                "total": int(len(df_res)),
                "sentiment": df_res["Sentiment"].value_counts().to_dict(),
                "urgency": df_res["Urgency"].value_counts().to_dict(),
                "product": df_res["Product"].value_counts().to_dict(),
                "topic": df_res["Topic"].value_counts().to_dict()
            }

            samples = df_res[["Feedback_Text", "Product", "Topic", "Sentiment", "Urgency"]].head(50).to_dict("records")

            prompt = f"""
Viết báo cáo quản trị bằng tiếng Việt từ dữ liệu VoC này.
Cấu trúc:
1. Tóm tắt điều hành
2. Phát hiện nổi bật
3. Rủi ro cần ưu tiên
4. Nguyên nhân có thể có
5. Khuyến nghị hành động
6. Hạn chế của dữ liệu

Không bịa số liệu. Phân biệt dữ kiện với giả thuyết.

THỐNG KÊ:
{json.dumps(stats, ensure_ascii=False)}

MẪU FEEDBACK:
{json.dumps(samples, ensure_ascii=False, default=str)}
"""

            # Retry loop xử lý lỗi 429 khi tạo báo cáo AI
            while True:
                try:
                    with st.spinner("Đang khởi tạo báo cáo quản trị bằng Gemini AI..."):
                        response = client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=prompt
                        )
                    st.session_state.voc_summary = response.text
                    st.success("✓ Đã sinh báo cáo thành công!")
                    break
                except Exception as error:
                    error_str = str(error)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        st.warning("⏳ Bị nghẽn Quota (429) ở Bước sinh báo cáo. Tạm dừng 35 giây...")
                        time.sleep(35)
                    else:
                        st.error(f"❌ Lỗi khi sinh báo cáo: {error}")
                        break

        if st.session_state.voc_summary:
            st.markdown("---")
            st.markdown(st.session_state.voc_summary)
            st.download_button(
                "⬇️ Tải báo cáo TXT",
                data=st.session_state.voc_summary,
                file_name="GoldLens_Management_Summary.txt",
                mime="text/plain"
            )

# --------------------------------------------------
# TRANG 5: TIN TỨC VIETSTOCK
# --------------------------------------------------
elif page == "Tin tức Vietstock":
    st.title("📰 Vietstock Financial News")
    st.write("Cập nhật thông tin thị trường tài chính, chứng khoán và kinh tế từ Vietstock RSS.")

    if st.button("🔄 Làm mới tin tức"):
        get_vietstock_news.clear()

    with st.spinner("Đang tải tin tức từ Vietstock..."):
        news = get_vietstock_news()

    if not news:
        st.warning("Không lấy được dữ liệu RSS từ Vietstock lúc này. Vui lòng truy cập trực tiếp:")
        st.markdown("[🔗 Mở Vietstock.vn](https://vietstock.vn/)")
    else:
        query = st.text_input("🔍 Tìm tin theo từ khóa:")
        limit = st.selectbox("Số lượng tin hiển thị:", [10, 20, 30, 50], index=1)

        if query:
            news = [item for item in news if query.lower() in (item["title"] + " " + item["description"]).lower()]

        st.caption(f"Tìm thấy {len(news)} tin tức.")

        for item in news[:limit]:
            st.markdown(f"### [{item['title']}]({item['link']})")
            if item["published"]:
                st.caption(f"Nguồn: Vietstock · {item['published']}")
            if item["description"]:
                st.write(item["description"][:500])
            st.divider()

st.sidebar.markdown("---")
st.sidebar.caption("GoldLens Finance · Streamlit App")
