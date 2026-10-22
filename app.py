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

from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ==================================================
# 1. CẤU HÌNH VÀ GIAO DIỆN
# ==================================================
st.set_page_config(
    page_title="GoldLens Finance",
    page_icon="🟡",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #FFFDF6;
    color: #302B1B;
}
[data-testid="stSidebar"] {
    background: #FFF3C4;
}
.hero {
    background: linear-gradient(120deg, #F4C430, #FFE99B, #FFF9E7);
    border: 1px solid #E8CC65;
    border-radius: 20px;
    padding: 26px 30px;
    margin-bottom: 20px;
    color: #302600;
}
.hero h1 {
    margin-bottom: 4px;
    font-size: 38px;
}
.hero p {
    font-size: 16px;
    margin-bottom: 0;
}
div.stButton > button,
div.stDownloadButton > button {
    background: #F4C430;
    color: #302600;
    border: 1px solid #D6B221;
    border-radius: 10px;
    font-weight: 700;
}
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #F0E2A0;
    border-radius: 14px;
    padding: 12px;
}
a {
    color: #8A6900 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🟡 GoldLens Finance</h1>
    <p>Customer Intelligence · Financial Insights · Market News</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# 2. CẤU HÌNH GEMINI
# ==================================================
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

@st.cache_resource
def get_gemini_client():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    return genai.Client(api_key=key)

client = get_gemini_client()

PRODUCTS = [
    "Mobile Banking", "Transfer", "ATM", "Credit Card",
    "Debit Card", "Loan", "Savings", "Branch Service", "Other"
]
TOPICS = [
    "Login", "Transaction", "App Performance", "Fee",
    "Customer Service", "Security", "ATM Failure",
    "Card", "Loan", "Interest Rate", "Other"
]
SENTIMENTS = ["Positive", "Neutral", "Negative"]
URGENCIES = ["Low", "Medium", "High", "Critical"]

# ==================================================
# 3. ĐỌC FILE EXCEL / CSV
# ==================================================
def find_column(df, candidates):
    columns = {
        str(c).strip().lower().replace(" ", "_"): c
        for c in df.columns
    }
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

    text_col = find_column(df, [
        "Feedback_Text", "feedback", "comment", "review",
        "content", "text", "customer_feedback", "noi_dung"
    ])

    if text_col is None:
        text_cols = list(
            df.select_dtypes(include=["object", "string"]).columns
        )
        if not text_cols:
            raise ValueError("Không tìm thấy cột nội dung feedback.")

        text_col = max(
            text_cols,
            key=lambda c: df[c].fillna("").astype(str).str.len().mean()
        )

    if text_col != "Feedback_Text":
        df["Feedback_Text"] = df[text_col]

    df["Feedback_Text"] = (
        df["Feedback_Text"].fillna("").astype(str).str.strip()
    )
    df = df[df["Feedback_Text"] != ""].copy()

    if df.empty:
        raise ValueError("File không có feedback hợp lệ.")

    if len(df) > 1000:
        raise ValueError(
            "Bản demo giới hạn 1.000 feedback mỗi lần. "
            "Hãy chia file thành các phần nhỏ hơn."
        )

    if "Feedback_ID" not in df.columns:
        df.insert(
            0, "Feedback_ID",
            [f"FB{i:05d}" for i in range(1, len(df) + 1)]
        )

    date_col = find_column(df, [
        "Feedback_Date", "date", "created_at",
        "review_date", "ngay", "ngày"
    ])

    if date_col and date_col != "Feedback_Date":
        df["Feedback_Date"] = df[date_col]
    elif "Feedback_Date" not in df.columns:
        df["Feedback_Date"] = pd.NaT

    df["Feedback_Date"] = pd.to_datetime(
        df["Feedback_Date"], errors="coerce"
    )

    rating_col = find_column(df, ["Rating", "score", "stars"])
    if rating_col and rating_col != "Rating":
        df["Rating"] = pd.to_numeric(
            df[rating_col], errors="coerce"
        )
    elif "Rating" not in df.columns:
        df["Rating"] = np.nan

    return df.reset_index(drop=True)

# ==================================================
# 4. PHÂN LOẠI BẰNG GEMINI
# ==================================================
def parse_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("Gemini không trả về JSON hợp lệ.")

def classify_one(feedback):
    if client is None:
        raise RuntimeError("Chưa cấu hình GEMINI_API_KEY.")

    prompt = f"""
Bạn là chuyên gia phân tích phản hồi khách hàng ngân hàng.
Hãy phân loại feedback theo các danh sách sau:

Product: {PRODUCTS}
Topic: {TOPICS}
Sentiment: {SENTIMENTS}
Urgency: {URGENCIES}

Critical: dấu hiệu mất tiền, gian lận hoặc rủi ro bảo mật nghiêm trọng.
High: giao dịch quan trọng thất bại hoặc dịch vụ ngừng hoạt động.
Medium: bất tiện nhưng khách vẫn sử dụng được.
Low: góp ý nhỏ hoặc lời khen.

Chỉ trả về một JSON object với đúng 4 khóa:
Product, Topic, Sentiment, Urgency.
Giữ nguyên ngôn ngữ của feedback khi suy luận ý nghĩa.
Không thêm giải thích.

Feedback:
{str(feedback)[:3000]}
"""

    allowed = {
        "Product": PRODUCTS,
        "Topic": TOPICS,
        "Sentiment": SENTIMENTS,
        "Urgency": URGENCIES
    }

    last_error = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )

            data = parse_json(response.text)
            result = {}

            for field, choices in allowed.items():
                value = str(data[field]).strip()
                if value not in choices:
                    raise ValueError(
                        f"Nhãn {field} không nằm trong danh sách."
                    )
                result[field] = value

            return result

        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(2 ** attempt)

    raise RuntimeError(str(last_error))

def classify_data(df):
    out = df.copy()

    for col in ["Product", "Topic", "Sentiment", "Urgency"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str).str.strip()

    progress = st.progress(0)
    status = st.empty()
    errors = []

    for i in range(len(out)):
        labels_exist = (
            out.at[i, "Product"] in PRODUCTS
            and out.at[i, "Topic"] in TOPICS
            and out.at[i, "Sentiment"] in SENTIMENTS
            and out.at[i, "Urgency"] in URGENCIES
        )

        if not labels_exist:
            try:
                labels = classify_one(out.at[i, "Feedback_Text"])
                for col, value in labels.items():
                    out.at[i, col] = value

            except Exception as error:
                errors.append({
                    "Feedback_ID": out.at[i, "Feedback_ID"],
                    "Feedback_Text": out.at[i, "Feedback_Text"],
                    "Error": str(error)
                })

        progress.progress((i + 1) / len(out))
        status.info(f"Đang xử lý feedback {i + 1}/{len(out)}")

    progress.empty()
    status.empty()

    return out, pd.DataFrame(errors)

# ==================================================
# 5. NATURAL TOPIC CLUSTERING
# ==================================================
def make_clusters(df, number_of_clusters=5):
    out = df.copy()
    texts = out["Feedback_Text"].fillna("").astype(str)

    if len(out) < 2:
        out["Cluster"] = 0
        out["Natural_Topic"] = "Nhóm feedback"
        return out

    k = min(max(2, int(number_of_clusters)), len(out))

    try:
        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2)
        )
        matrix = vectorizer.fit_transform(texts)

        labels = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        ).fit_predict(matrix)

        out["Cluster"] = labels
        terms = np.array(vectorizer.get_feature_names_out())
        names = {}

        for cluster in sorted(set(labels)):
            mask = labels == cluster
            mean_vector = np.asarray(
                matrix[mask].mean(axis=0)
            ).ravel()
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

# ==================================================
# 6. XUẤT EXCEL
# ==================================================
def export_excel(df, errors):
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(
            writer, index=False, sheet_name="VoC_Results"
        )
        if not errors.empty:
            errors.to_excel(
                writer, index=False, sheet_name="Errors"
            )

    return buffer.getvalue()

# ==================================================
# 7. TIN TỨC VIETSTOCK QUA RSS
# ==================================================
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
            response = requests.get(
                url,
                timeout=12,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)

            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                description = (
                    item.findtext("description") or ""
                )
                published = (item.findtext("pubDate") or "").strip()

                description = re.sub(
                    r"<[^>]+>", " ", description
                )
                description = re.sub(
                    r"\s+", " ", description
                ).strip()

                if title and link:
                    items.append({
                        "title": title,
                        "link": link,
                        "description": description,
                        "published": published
                    })

        except Exception:
            continue

    unique = {}
    for item in items:
        unique[item["link"]] = item

    return list(unique.values())

# ==================================================
# 8. MENU VÀ TRẠNG THÁI
# ==================================================
if "voc_df" not in st.session_state:
    st.session_state.voc_df = None

if "voc_errors" not in st.session_state:
    st.session_state.voc_errors = pd.DataFrame()

if "voc_summary" not in st.session_state:
    st.session_state.voc_summary = ""

page = st.sidebar.radio(
    "GOLDLENS MENU",
    [
        "Trang chủ",
        "Phân tích khách hàng",
        "Dashboard",
        "Báo cáo quản trị AI",
        "Tin tức Vietstock"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("GOLDLENS FINANCE")

if client is None:
    st.sidebar.warning(
        "Gemini chưa được cấu hình. "
        "Hãy thêm GEMINI_API_KEY trong Render Environment."
    )
else:
    st.sidebar.success("Gemini đã sẵn sàng")

# ==================================================
# 9. TRANG CHỦ
# ==================================================
if page == "Trang chủ":
    st.title("Chào mừng đến với GoldLens Finance")

    st.write(
        "Nền tảng phân tích Voice of Customer kết hợp "
        "thông tin thị trường tài chính."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 🧠 Customer Intelligence")
        st.write(
            "Phân loại cảm xúc, sản phẩm, chủ đề "
            "và mức độ khẩn cấp."
        )

    with c2:
        st.markdown("### 📊 Smart Dashboard")
        st.write(
            "Theo dõi phản hồi, nhóm chủ đề "
            "và các vấn đề cần ưu tiên."
        )

    with c3:
        st.markdown("### 📰 Vietstock News")
        st.write(
            "Theo dõi tiêu đề và liên kết tin tài chính "
            "khi nguồn RSS khả dụng."
        )

    st.info(
        "Bắt đầu bằng cách mở mục Phân tích khách hàng "
        "và tải file Excel hoặc CSV."
    )

# ==================================================
# 10. PHÂN TÍCH KHÁCH HÀNG
# ==================================================
elif page == "Phân tích khách hàng":
    st.title("Phân tích Voice of Customer")

    uploaded = st.file_uploader(
        "Tải file feedback",
        type=["xlsx", "xls", "csv"]
    )

    number_of_clusters = st.slider(
        "Số nhóm Natural Topic",
        min_value=2,
        max_value=10,
        value=5
    )

    if uploaded is not None:
        try:
            if uploaded.size > 15 * 1024 * 1024:
                raise ValueError("File phải nhỏ hơn 15 MB.")

            preview = load_feedback(uploaded)

            st.success(
                f"Đọc được {len(preview)} feedback."
            )
            st.dataframe(
                preview.head(10),
                use_container_width=True
            )

            if st.button(
                "🚀 Chạy phân tích",
                type="primary"
            ):
                with st.spinner("Đang phân loại bằng Gemini..."):
                    result, errors = classify_data(preview)

                with st.spinner("Đang gom nhóm chủ đề..."):
                    result = make_clusters(
                        result, number_of_clusters
                    )

                result["Month"] = (
                    result["Feedback_Date"]
                    .dt.to_period("M")
                    .astype(str)
                )

                st.session_state.voc_df = result
                st.session_state.voc_errors = errors
                st.session_state.voc_summary = ""

                st.success("Đã xử lý xong!")

        except Exception as error:
            st.error(f"Không thể đọc hoặc xử lý file: {error}")

    df = st.session_state.voc_df

    if df is not None:
        st.subheader("Kết quả phân tích")

        query = st.text_input(
            "Tìm kiếm trong dữ liệu kết quả"
        )

        display_df = df.copy()

        if query:
            mask = display_df.astype(str).apply(
                lambda col: col.str.contains(
                    query, case=False, na=False
                )
            ).any(axis=1)
            display_df = display_df[mask]

        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )

        st.download_button(
            "⬇️ Tải kết quả Excel",
            data=export_excel(
                df, st.session_state.voc_errors
            ),
            file_name="GoldLens_VoC_Results.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        )

        if not st.session_state.voc_errors.empty:
            st.warning(
                f"{len(st.session_state.voc_errors)} feedback "
                "chưa phân loại thành công. Xem sheet Errors "
                "trong file Excel."
            )

# ==================================================
# 11. DASHBOARD
# ==================================================
elif page == "Dashboard":
    st.title("Customer Intelligence Dashboard")

    df = st.session_state.voc_df

    if df is None:
        st.info("Hãy phân tích một file feedback trước.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Tổng feedback", f"{len(df):,}")
        c2.metric(
            "Tiêu cực",
            int((df["Sentiment"] == "Negative").sum())
        )
        c3.metric(
            "Ưu tiên cao",
            int(df["Urgency"].isin(["High", "Critical"]).sum())
        )
        c4.metric(
            "Nhóm chủ đề",
            int(df["Natural_Topic"].nunique())
        )

        sentiment = (
            df["Sentiment"].value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Count")
        )

        st.plotly_chart(
            px.pie(
                sentiment,
                names="Sentiment",
                values="Count",
                title="Phân bố cảm xúc",
                hole=0.4
            ),
            use_container_width=True
        )

        left, right = st.columns(2)

        with left:
            product = (
                df["Product"].value_counts()
                .rename_axis("Product")
                .reset_index(name="Count")
            )

            st.plotly_chart(
                px.bar(
                    product,
                    x="Product",
                    y="Count",
                    title="Feedback theo sản phẩm"
                ),
                use_container_width=True
            )

        with right:
            topic = (
                df["Topic"].value_counts()
                .rename_axis("Topic")
                .reset_index(name="Count")
            )

            st.plotly_chart(
                px.bar(
                    topic,
                    x="Topic",
                    y="Count",
                    title="Feedback theo chủ đề"
                ),
                use_container_width=True
            )

        st.subheader("🚨 Cảnh báo sớm")

        urgent = df[
            df["Urgency"].isin(["High", "Critical"])
            | (df["Sentiment"] == "Negative")
        ]

        if urgent.empty:
            st.success(
                "Không phát hiện feedback tiêu cực/khẩn cấp "
                "theo nhãn hiện tại."
            )
        else:
            st.warning(
                f"Có {len(urgent)} feedback cần rà soát."
            )

            st.dataframe(
                urgent[
                    [
                        "Feedback_ID", "Feedback_Text",
                        "Product", "Topic",
                        "Sentiment", "Urgency"
                    ]
                ],
                use_container_width=True
            )

        dated = df.dropna(subset=["Feedback_Date"]).copy()

        if not dated.empty:
            dated["Date"] = (
                dated["Feedback_Date"].dt.date.astype(str)
            )

            trend = (
                dated.groupby(["Date", "Sentiment"])
                .size()
                .reset_index(name="Count")
            )

            st.plotly_chart(
                px.line(
                    trend,
                    x="Date",
                    y="Count",
                    color="Sentiment",
                    markers=True,
                    title="Xu hướng feedback theo ngày"
                ),
                use_container_width=True
            )
        else:
            st.info(
                "Muốn xem xu hướng, hãy thêm cột ngày vào file."
            )

        st.subheader("Natural Topic Clustering")

        natural = (
            df["Natural_Topic"].value_counts()
            .rename_axis("Natural_Topic")
            .reset_index(name="Count")
        )

        st.dataframe(natural, use_container_width=True)

# ==================================================
# 12. BÁO CÁO QUẢN TRỊ AI
# ==================================================
elif page == "Báo cáo quản trị AI":
    st.title("AI Management Summary")

    df = st.session_state.voc_df

    if df is None:
        st.info("Hãy phân tích feedback trước.")
    elif client is None:
        st.warning(
            "Cần cấu hình GEMINI_API_KEY để tạo báo cáo."
        )
    else:
        st.write(
            "Báo cáo dựa trên số liệu phân loại và mẫu feedback. "
            "Không gửi thông tin định danh khách hàng."
        )

        if st.button(
            "✨ Tạo báo cáo quản trị",
            type="primary"
        ):
            stats = {
                "total": int(len(df)),
                "sentiment": df["Sentiment"].value_counts().to_dict(),
                "urgency": df["Urgency"].value_counts().to_dict(),
                "product": df["Product"].value_counts().to_dict(),
                "topic": df["Topic"].value_counts().to_dict()
            }

            samples = df[
                [
                    "Feedback_Text", "Product",
                    "Topic", "Sentiment", "Urgency"
                ]
            ].head(80).to_dict("records")

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
Không khẳng định nguyên nhân khi feedback chưa đủ bằng chứng.

THỐNG KÊ:
{json.dumps(stats, ensure_ascii=False)}

MẪU FEEDBACK:
{json.dumps(samples, ensure_ascii=False, default=str)}
"""

            try:
                with st.spinner("Đang tạo báo cáo..."):
                    response = client.models.generate_content(
                        model=MODEL,
                        contents=prompt
                    )

                st.session_state.voc_summary = response.text

            except Exception as error:
                st.error(f"Lỗi tạo báo cáo: {error}")

        if st.session_state.voc_summary:
            st.markdown(st.session_state.voc_summary)

            st.download_button(
                "Tải báo cáo TXT",
                data=st.session_state.voc_summary,
                file_name="GoldLens_Management_Summary.txt",
                mime="text/plain"
            )

# ==================================================
# 13. TIN TỨC VIETSTOCK
# ==================================================
elif page == "Tin tức Vietstock":
    st.title("📰 Vietstock Financial News")

    st.write(
        "Tin tài chính, chứng khoán, doanh nghiệp và kinh tế."
    )

    if st.button("🔄 Làm mới tin"):
        get_vietstock_news.clear()

    with st.spinner("Đang tải tin từ RSS Vietstock..."):
        news = get_vietstock_news()

    if not news:
        st.warning(
            "Không lấy được dữ liệu RSS lúc này. "
            "Nguồn có thể tạm thời không phản hồi hoặc thay đổi."
        )

        st.markdown(
            "[Mở Vietstock](https://vietstock.vn/)"
        )
        st.markdown(
            "[Mở VietstockFinance](https://finance.vietstock.vn/)"
        )

    else:
        query = st.text_input("Tìm tin theo từ khóa")
        limit = st.selectbox(
            "Số tin muốn hiển thị",
            [10, 20, 30, 50],
            index=1
        )

        if query:
            news = [
                item for item in news
                if query.lower() in (
                    item["title"] + " " + item["description"]
                ).lower()
            ]

        st.caption(f"Tìm thấy {len(news)} tin.")

        for item in news[:limit]:
            st.markdown(f"### [{item['title']}]({item['link']})")

            if item["published"]:
                st.caption(f"Nguồn: Vietstock · {item['published']}")

            if item["description"]:
                st.write(item["description"][:500])

            st.divider()

st.sidebar.markdown("---")
st.sidebar.caption("GoldLens Finance · Powered by Streamlit")
