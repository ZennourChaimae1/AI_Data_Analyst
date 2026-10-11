import os
import io
import json
import re
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from groq import Groq
from supabase import create_client


from database import (
    init_db,
    create_user,
    authenticate_user,
    get_user_id,
    add_dataset
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.cluster import KMeans
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
)


# =========================================================
# PROFESSIONAL DARK UI + LUFFY BACKGROUND
# =========================================================

import base64
from pathlib import Path

# Load Luffy background image if available. The app still works without it.
image_path = Path(__file__).parent / "Luffy.jpeg"
luffy_bg = ""
if image_path.is_file():
    try:
        with open(image_path, "rb") as image_file:
            luffy_bg = base64.b64encode(image_file.read()).decode("utf-8")
    except OSError:
        luffy_bg = ""

background_css = (
    f'background-image: linear-gradient(rgba(10, 10, 14, 0.72), rgba(10, 10, 14, 0.82)), '
    f'url("data:image/jpeg;base64,{luffy_bg}");'
    if luffy_bg else
    'background-image: linear-gradient(rgba(10, 10, 14, 0.72), rgba(10, 10, 14, 0.82));'
)


st.markdown(f"""
<style>

    /* Main app */
    .stApp {{
        background-color: #0f1117;
        {background_css}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #f8fafc;
    }}

    /* Main content */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}

    /* Titles */
    h1, h2, h3, h4 {{
        color: #f8fafc !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    /* Text */
    p, label, span {{
        color: #e5e7eb;
    }}

    /* Cards */
    .ui-card {{
        background: rgba(23, 26, 33, 0.88);
        padding: 1.25rem;
        border-radius: 14px;
        border: 1px solid #2a2f3a;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
        backdrop-filter: blur(6px);
    }}

    /* KPI */
    .kpi-card {{
        background: rgba(23, 26, 33, 0.88);
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid #2a2f3a;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
        backdrop-filter: blur(6px);
    }}

    .kpi-title {{
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }}

    .kpi-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
        border: 1px solid #343a46;
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        border-color: #a5b4fc;
        transform: translateY(-1px);
    }}

    /* Inputs */
    .stTextInput input,
    .stNumberInput input {{
        background-color: rgba(23, 26, 33, 0.94) !important;
        color: #f8fafc !important;
        border: 1px solid #343a46 !important;
        border-radius: 10px;
    }}

    /* Selectbox */
    div[data-baseweb="select"] > div {{
        background-color: rgba(23, 26, 33, 0.94) !important;
        border-color: #343a46 !important;
        border-radius: 10px;
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] {{
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2a2f3a;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: rgba(23, 26, 33, 0.96);
        border-right: 1px solid #343741;
    }}

    /* Alerts */
    .stAlert {{
        border-radius: 10px;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] {{
        background-color: rgba(23, 26, 33, 0.90);
        border-radius: 12px;
        border: 1px solid #2a2f3a;
    }}

</style>
""", unsafe_allow_html=True)


# SUPABASE
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL", SUPABASE_URL)
    SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", SUPABASE_KEY)
except Exception:
    # Streamlit secrets may be unavailable in local development; use .env values.
    pass

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None

# GROQ
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", GROQ_API_KEY)
    GROQ_MODEL = st.secrets.get("GROQ_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
except Exception:
    # Local development can use values from .env.
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)
init_db()

# =========================================================
# AUTHENTICATION
# =========================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

def safe_username(username):
    username = username.strip()

    if not username:
        return False

    if username in {".", ".."}:
        return False

    if os.path.basename(username) != username:
        return False

    if any(char in username for char in '<>:"/\\|?*'):
        return False

    return True

if not st.session_state.authenticated:
    st.title("🤖 AI Data Analyst")
    st.caption("Analyze, visualize and explore your datasets with AI.")

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Create Account"])

    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", use_container_width=True):
            clean_username = username.strip()
            login_ok = authenticate_user(clean_username, password)    

            if login_ok:
                st.session_state.authenticated = True
                st.session_state.username = clean_username
                st.rerun()
            else:
                st.error("Username or password incorrect.")
    
    

    with tab2:
        new_username = st.text_input("Username", key="register_username").strip()
        new_email = st.text_input("Email", key="register_email")
        new_password = st.text_input(
            "Password",
            type="password",
            key="register_password"
        )
        confirm_password = st.text_input(
            "Confirm password",
            type="password",
            key="register_confirm"
        )

        if st.button("Create Account", use_container_width=True):
            if not new_username or not new_email or not new_password:
                st.warning("Please fill in all fields.")

            elif new_password != confirm_password:
                st.error("Passwords do not match.")

            elif not safe_username(new_username):
                st.error("Invalid username.")

            else:
                if create_user(new_username, new_email, new_password):
                    st.success("Account created successfully. You can now login.")
                else:
                    st.error("Username or email already exists.")

    st.stop()
    

# =========================================================
# USER WORKSPACE - SUPABASE STORAGE
# =========================================================
# =========================================================
# PROFESSIONAL SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🤖 AI Data Analyst")
st.sidebar.caption("Intelligent Data Analysis")
st.sidebar.markdown("---")
st.sidebar.caption("WORKSPACE")

page = st.sidebar.radio(
    "Navigation",
    [
        "📂 My Datasets",
        "🧹 Data Cleaning",
        "📊 Dashboard",
        "📈 Statistics & EDA",
        "🤖 AI Insights",
        "🧠 AI Agent",
        "💬 AI Chat",
        "📄 Reports"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("AI Data Analyst")
with st.sidebar:
    st.title("🤖 AI Data Analyst")
    st.write(f"👤 **{st.session_state.username}**")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.pop("selected_file", None)
        st.session_state.pop("selected_df", None)
        st.session_state.pop("cleaned_df", None)
        st.session_state.pop("chat_history", None)
        st.session_state.pop("chat_context", None)
        st.rerun()

# =========================================================
# HELPERS
# =========================================================


def get_user_storage_folder():
    username = st.session_state.get("username")

    if not username:
        raise ValueError("User is not logged in.")

    return str(username)

def list_user_files():
    if supabase is None:
        raise RuntimeError("Supabase is not configured.")

    folder = get_user_storage_folder()

    try:
        files = supabase.storage.from_("datasets").list(
            path=folder,
            options={"limit": 100}
        )

        return sorted([
            f["name"]
            for f in files
            if f.get("name", "").lower().endswith(
                (".csv", ".xlsx", ".xls")
            )
        ])

    except Exception as e:
        st.error(
            f"Storage error: {type(e).__name__} - {str(e)}"
        )
        return []

def read_dataset(file_name):
    if supabase is None:
        raise RuntimeError("Supabase is not configured.")

    folder = get_user_storage_folder()
    file_path = f"{folder}/{file_name}"

    file_bytes = supabase.storage.from_("datasets").download(file_path)
    buffer = io.BytesIO(file_bytes)

    if file_name.lower().endswith(".csv"):
        return pd.read_csv(buffer)

    return pd.read_excel(buffer)


def set_selected_dataset(file_name):
    data = read_dataset(file_name)

    st.session_state.selected_file = file_name
    st.session_state.selected_df = data.copy()
    st.session_state.cleaned_df = data.copy()
    st.session_state.cleaned_from_file = file_name
    st.session_state.chat_history = []
    st.session_state.chat_context = {}
    st.session_state.agent_results = []
    st.session_state.pop("ai_insights", None)
    st.session_state.pop("final_report", None)


def get_working_df():
    if "selected_file" not in st.session_state:
        return None

    if st.session_state.get("cleaned_from_file") != st.session_state.selected_file:
        set_selected_dataset(st.session_state.selected_file)

    return st.session_state.get("cleaned_df")


def save_working_df():
    if (
        "selected_file" not in st.session_state
        or "cleaned_df" not in st.session_state
    ):
        return False

    if supabase is None:
        st.error("Supabase is not configured.")
        return False

    folder = get_user_storage_folder()
    file_name = st.session_state.selected_file
    file_path = f"{folder}/{file_name}"
    df = st.session_state.cleaned_df

    try:
        if file_name.lower().endswith(".csv"):
            file_bytes = df.to_csv(index=False).encode("utf-8")
            content_type = "text/csv"
        else:
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            file_bytes = buffer.getvalue()
            content_type = (
                "application/vnd.openxmlformats-officedocument"
                ".spreadsheetml.sheet"
            )

        supabase.storage.from_("datasets").upload(
            file_path,
            file_bytes,
            file_options={
                "upsert": "true",
                "content-type": content_type,
            },
        )

        st.session_state.selected_df = df.copy()
        return True

    except Exception as e:
        st.error(f"Could not save dataset: {e}")
        return False


def dataset_context(df, sample_rows=8):
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "numeric": numeric,
        "categorical": categorical,
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "sample": df.head(sample_rows).to_string(index=False),
    }


def safe_json(text):
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    for i, char in enumerate(text):
        if char != "{":
            continue

        try:
            obj, _ = decoder.raw_decode(text[i:])

            if isinstance(obj, dict):
                return obj

        except json.JSONDecodeError:
            continue

    return None


def language_instruction(language):
    """Return a clear language instruction for AI-generated analysis content."""
    if language == "Auto (same as AI Chat)":
        history = st.session_state.get("chat_history", [])
        last_user_message = next(
            (item.get("content", "") for item in reversed(history) if item.get("role") == "user"),
            "",
        )
        if last_user_message:
            return (
                "Write the whole response in the same language as this latest user message from AI Chat: "
                f"{last_user_message!r}. Keep dataset column names unchanged."
            )
        return "Use French because no prior user message from AI Chat is available. Keep dataset column names unchanged."
    language_names = {
        "Français": "French (français)",
        "English": "English",
        "العربية": "Arabic (العربية)",
        "Español": "Spanish (español)",
    }
    chosen = language_names.get(language, "French (français)")
    return f"Write the entire response in {chosen}. Do not switch to another language. Keep dataset column names exactly as they appear in the data."


def ai_chat(prompt):
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def make_bar_or_line(df, x, y, chart_type, title):
    if x not in df.columns or y not in df.columns:
        raise ValueError("The selected chart columns do not exist in the dataset.")
    if y not in numeric_columns(df):
        raise ValueError("Bar and line charts require a numeric Y column.")
    data = df.groupby(x, dropna=False, observed=False)[y].sum().reset_index()
    if chart_type == "line":
        return px.line(data, x=x, y=y, title=title, markers=True)
    return px.bar(data, x=x, y=y, title=title)


def create_pdf(df, report_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading = ParagraphStyle("Heading", parent=styles["Heading2"], spaceBefore=8, spaceAfter=5)
    body = ParagraphStyle("Body", parent=styles["BodyText"], leading=14, spaceAfter=5)

    story = [Paragraph("AI Data Analysis Report", title_style), Spacer(1, 0.3*cm)]
    summary = [
        ["Rows", str(len(df))],
        ["Columns", str(len(df.columns))],
        ["Missing values", str(int(df.isna().sum().sum()))],
        ["Duplicates", str(int(df.duplicated().sum()))],
    ]
    table = Table(summary, colWidths=[6*cm, 8*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [Paragraph("Dataset Overview", heading), table, Spacer(1, 0.3*cm)]

    for line in report_text.splitlines():
        line = line.strip()
        if not line:
            continue
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if re.match(r"^[1-6][.)]", line):
            story.append(Paragraph(safe, heading))
        else:
            story.append(Paragraph(safe, body))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# PAGE CONTENT
# =========================================================

if page == "📂 My Datasets":
    
    # =========================================================
    # 1. UPLOAD DATA
    # =========================================================
    st.header("📂 1. Upload your data")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:

        safe_name = os.path.basename(uploaded_file.name)

        if safe_name != uploaded_file.name:
            st.error("Invalid file name.")
            st.stop()

        try:
            if supabase is None:
                st.error("Supabase is not configured.")
                st.stop()

            # Get current user's ID
            username = st.session_state.username
            user_id = get_user_id(username)

            if user_id is None:
                st.error("User not found in database.")
                st.stop()

            # Read uploaded file into memory
            file_bytes = uploaded_file.getvalue()

            if safe_name.lower().endswith(".csv"):
                df = pd.read_csv(io.BytesIO(file_bytes))
                content_type = "text/csv"
            else:
                df = pd.read_excel(io.BytesIO(file_bytes))
                content_type = (
                    "application/vnd.openxmlformats-officedocument"
                    ".spreadsheetml.sheet"
                )

            # Upload to the user's folder in Supabase Storage
            storage_path = f"{username}/{safe_name}"

            supabase.storage.from_("datasets").upload(
                storage_path,
                file_bytes,
                file_options={
                    "upsert": "true",
                    "content-type": content_type,
                },
            )

            # Select dataset in the app
            st.session_state.selected_file = safe_name
            st.session_state.selected_df = df.copy()
            st.session_state.cleaned_df = df.copy()
            st.session_state.cleaned_from_file = safe_name
            st.session_state.chat_history = []
            st.session_state.chat_context = {}
            st.session_state.agent_results = []
            st.session_state.pop("ai_insights", None)
            st.session_state.pop("final_report", None)

            # Keep dataset metadata in PostgreSQL
            relative_path = storage_path

            saved = add_dataset(
                user_id,
                safe_name,
                relative_path
            )

            if saved:
                st.success(
                    f"✅ File uploaded and registered: {safe_name}"
                )
            else:
                st.warning(
                    "File uploaded to Supabase, but its metadata "
                    "could not be registered in PostgreSQL."
                )

            st.rerun()

        except Exception as e:
            st.error(f"Error while uploading the file: {e}")

    # =========================================================
    # 2. MY DATASETS + SELECT DATASET
    # =========================================================

    st.header("📁 My Datasets")

    files = list_user_files()

    if files:
        for file_name in files:
            storage_path = f"{get_user_storage_folder()}/{file_name}"

            c1, c2, c3 = st.columns([5, 1, 1])

            with c1:
                st.write(f"📄 {file_name}")

            with c2:
                try:
                    file_bytes = (
                        supabase.storage
                        .from_("datasets")
                        .download(storage_path)
                    )

                    st.download_button(
                        "⬇️",
                        data=file_bytes,
                        file_name=file_name,
                        key=f"download_{file_name}"
                    )

                except Exception as e:
                    st.error(f"Download failed: {e}")

            with c3:
                if st.button("✏️", key=f"rename_{file_name}"):
                    st.session_state[f"rename_mode_{file_name}"] = True

                if st.button("🗑️", key=f"delete_{file_name}"):
                    try:
                        supabase.storage.from_("datasets").remove(
                            [storage_path]
                        )

                        if st.session_state.get("selected_file") == file_name:
                            for key in [
                                "selected_file",
                                "selected_df",
                                "cleaned_df",
                                "cleaned_from_file",
                            ]:
                                st.session_state.pop(key, None)

                        st.session_state.pop(
                            f"rename_mode_{file_name}", None
                        )

                        st.success(f"Deleted: {file_name}")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Delete failed: {e}")

            if st.session_state.get(f"rename_mode_{file_name}", False):
                new_name = st.text_input(
                    "New dataset name",
                    value=os.path.splitext(file_name)[0],
                    key=f"new_name_{file_name}"
                ).strip()

                rc1, rc2 = st.columns(2)

                with rc1:
                    if st.button("💾 Save", key=f"save_rename_{file_name}"):
                        if not new_name:
                            st.warning("Please enter a name.")
                        elif new_name in (".", "..") or "/" in new_name or "\\" in new_name:
                            st.error("Invalid dataset name.")
                        else:
                            extension = os.path.splitext(file_name)[1]
                            new_file = new_name + extension
                            new_path = (
                                f"{get_user_storage_folder()}/{new_file}"
                            )

                            if new_file in files:
                                st.error(
                                    "A file with this name already exists."
                                )
                            else:
                                try:
                                    file_bytes = (
                                        supabase.storage
                                        .from_("datasets")
                                        .download(storage_path)
                                    )

                                    content_type = (
                                        "text/csv"
                                        if file_name.lower().endswith(".csv")
                                        else "application/octet-stream"
                                    )

                                    supabase.storage.from_("datasets").upload(
                                        new_path,
                                        file_bytes,
                                        file_options={
                                            "upsert": "false",
                                            "content-type": content_type,
                                        },
                                    )

                                    supabase.storage.from_("datasets").remove(
                                        [storage_path]
                                    )

                                    if st.session_state.get("selected_file") == file_name:
                                        st.session_state.selected_file = new_file
                                        st.session_state.cleaned_from_file = new_file

                                    st.session_state.pop(
                                        f"rename_mode_{file_name}", None
                                    )

                                    st.rerun()

                                except Exception as e:
                                    st.error(f"Rename failed: {e}")

                with rc2:
                    if st.button("Cancel", key=f"cancel_rename_{file_name}"):
                        st.session_state.pop(
                            f"rename_mode_{file_name}", None
                        )
                        st.rerun()

        selected_file = st.selectbox(
            "Choose a dataset to work with:",
            files,
            index=(
                files.index(st.session_state["selected_file"])
                if st.session_state.get("selected_file") in files
                else 0
            ),
            key="dataset_selector"
        )

        if st.session_state.get("selected_file") != selected_file:
            set_selected_dataset(selected_file)

    else:
        st.info("Upload a dataset first.")


    # Refresh active dataset after upload/selection changes.
    working_df = get_working_df()
    num_cols = (
        numeric_columns(working_df)
        if working_df is not None
        else []
    )
    cat_cols = (
        categorical_columns(working_df)
        if working_df is not None
        else []
    )

    if page == "📂 My Datasets":
        if working_df is None:
            st.info("Upload a dataset first.")
        else:
            st.success(
                f"✅ Active dataset: {st.session_state.selected_file}"
            )

    # Refresh active dataset
      
    working_df = get_working_df()

    num_cols = numeric_columns(working_df) if working_df is not None else []
    cat_cols = categorical_columns(working_df) if working_df is not None else [] 

    
    # =========================================================
    # 3. DATASET PREVIEW & INFORMATION
    # =========================================================

    st.header("👀 3. Dataset Preview")

    if working_df is not None:
        st.dataframe(
            working_df.head(10),
            use_container_width=True
        )

        st.header("📋 4. Dataset Information")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", len(working_df))
        c2.metric("Columns", len(working_df.columns))
        c3.metric(
            "Missing Values",
            int(working_df.isna().sum().sum())
        )
        c4.metric(
            "Duplicates",
            int(working_df.duplicated().sum())
        )

        columns_info = pd.DataFrame({
            "Column": working_df.columns,
            "Data Type": working_df.dtypes.astype(str).values,
            "Missing Values": working_df.isna().sum().values,
            "Unique Values": working_df.nunique().values,
        })

        st.dataframe(columns_info, use_container_width=True)

    else:
        st.info("Upload or select a dataset to preview it.")



# Refresh active dataset before using it
working_df = get_working_df()

num_cols = (
    numeric_columns(working_df)
    if working_df is not None
    else []
)

cat_cols = (
    categorical_columns(working_df)
    if working_df is not None
    else []
)

if page != "📂 My Datasets" and working_df is None:
    st.warning("No dataset selected.")
    st.info(
        "Go to 📂 My Datasets, upload or select a dataset, "
        "then return to this page."
    )
    st.stop()

if page == "🧹 Data Cleaning":
    # =========================================================
    # 5. DATA QUALITY & CLEANING
    # =========================================================

    st.title("🧹 Data Cleaning")
    st.caption("Inspect data quality and clean your dataset.")

    # =========================================================
    # DATA QUALITY OVERVIEW
    # =========================================================

    missing_before = int(working_df.isna().sum().sum())
    duplicates_before = int(working_df.duplicated().sum())

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Missing Values",
        f"{missing_before:,}"
    )

    q2.metric(
        "Duplicate Rows",
        f"{duplicates_before:,}"
    )

    q3.metric(
        "Total Rows",
        f"{len(working_df):,}"
    )

    st.markdown("---")

    # =========================================================
    # CLEANING ACTIONS
    # =========================================================

    st.subheader("🛠️ Cleaning Actions")

    clean_col1, clean_col2, clean_col3 = st.columns(3)

    with clean_col1:
        if st.button(
            "🗑️ Remove duplicates",
            use_container_width=True
        ):
            before = len(working_df)

            st.session_state.cleaned_df = (
                working_df.drop_duplicates().copy()
            )

            removed = (
                before -
                len(st.session_state.cleaned_df)
            )

            st.success(
                f"Removed {removed} duplicate rows."
            )

            st.rerun()

    with clean_col2:
        if st.button(
            "🔢 Fill numeric missing",
            use_container_width=True
        ):
            df2 = working_df.copy()

            for col in numeric_columns(df2):
                if df2[col].isna().any():
                    df2[col] = df2[col].fillna(
                        df2[col].median()
                    )

            st.session_state.cleaned_df = df2

            st.success(
                "Numeric missing values filled with median."
            )

            st.rerun()

    with clean_col3:
        if st.button(
            "↩️ Reload original",
            use_container_width=True
        ):
            set_selected_dataset(
                st.session_state.selected_file
            )

            st.success(
                "Original dataset restored."
            )

            st.rerun()

    st.markdown("---")

    # =========================================================
    # SAVE CHANGES
    # =========================================================

    st.subheader("💾 Save Changes")

    if st.button(
        "💾 Save cleaning changes to dataset",
        use_container_width=True
    ):
        if save_working_df():
            st.success("Cleaning changes saved.")
        else:
            st.error("Cleaning changes were not saved. Check the error above and try again.")

if page == "📈 Statistics & EDA":
    # =========================================================
    # 6. STATISTICS & EDA
    # =========================================================

    st.title("📈 Statistics & EDA")
    st.caption("Explore statistical summaries, distributions, correlations and outliers.")

    num_cols = numeric_columns(working_df)
    cat_cols = categorical_columns(working_df)

    # =========================================================
    # STATISTICAL SUMMARY
    # =========================================================

    if num_cols:

        st.subheader("📊 Statistical Summary")

        st.dataframe(
            working_df[num_cols].describe().T,
            use_container_width=True
        )

        # =====================================================
        # DISTRIBUTIONS
        # =====================================================

        st.subheader("📉 Distributions")

        eda_col1, eda_col2 = st.columns(2)

        with eda_col1:

            hist_col = st.selectbox(
                "Histogram variable",
                num_cols,
                key="eda_hist_col"
            )

            st.plotly_chart(
                px.histogram(
                    working_df,
                    x=hist_col,
                    title=f"Distribution of {hist_col}"
                ),
                use_container_width=True
            )

        with eda_col2:

            box_col = st.selectbox(
                "Box plot variable",
                num_cols,
                key="eda_box_col"
            )

            st.plotly_chart(
                px.box(
                    working_df,
                    y=box_col,
                    title=f"Box Plot of {box_col}"
                ),
                use_container_width=True
            )

        # =====================================================
        # CORRELATION
        # =====================================================

        if len(num_cols) >= 2:

            st.subheader("🔗 Correlation Matrix")

            st.plotly_chart(
                px.imshow(
                    working_df[num_cols].corr(),
                    text_auto=True,
                    aspect="auto"
                ),
                use_container_width=True
            )

            # =================================================
            # OUTLIERS
            # =================================================

            outliers = []

            for col in num_cols:

                s = working_df[col].dropna()

                if len(s) < 4:
                    continue

                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1

                count = int(
                    (
                        (s < q1 - 1.5 * iqr)
                        |
                        (s > q3 + 1.5 * iqr)
                    ).sum()
                )

                if count:
                    outliers.append(
                        {
                            "Column": col,
                            "Outliers": count
                        }
                    )

            if outliers:

                st.subheader("⚠️ Potential Outliers")

                st.dataframe(
                    pd.DataFrame(outliers),
                    use_container_width=True
                )

    else:

        st.info(
            "No numeric columns available for numerical EDA."
        )

if page == "📊 Dashboard":
    # =========================================================
    # 7. PROFESSIONAL DASHBOARD
    # =========================================================

    st.title("📊 Dashboard")
    st.caption("Explore your dataset, KPIs and interactive visualizations.")

    dashboard_df = working_df.copy()

    # =========================================================
    # FILTERS
    # =========================================================

    filter_cols = [
        c for c in cat_cols
        if dashboard_df[c].nunique(dropna=False) <= 50
    ]

    if filter_cols:
        st.subheader("🎛️ Filters")

        fcols = st.columns(min(3, len(filter_cols)))

        for i, col in enumerate(filter_cols[:3]):
            values = dashboard_df[col].dropna().unique().tolist()

            choice = fcols[i].multiselect(
                col,
                values,
                default=[],
                key=f"dashboard_filter_{col}_{i}"
            )

            if choice:
                dashboard_df = dashboard_df[
                    dashboard_df[col].isin(choice)
                ]

    # =========================================================
    # KPIs
    # =========================================================

    kpi_cols = numeric_columns(dashboard_df)

    if kpi_cols:
        kcols = st.columns(min(4, len(kpi_cols)))

        for i, col in enumerate(kpi_cols[:4]):
            kcols[i].metric(
                f"Total {col}",
                f"{dashboard_df[col].sum():,.2f}"
            )

    # =========================================================
    # CHARTS
    # =========================================================

    if len(dashboard_df) == 0:
        st.warning("No data available after applying filters.")

    else:
        chart_options = [
            "Bar Chart",
            "Line Chart",
            "Scatter Plot",
            "Histogram",
            "Pie Chart",
            "Box Plot"
        ]

        chart_type = st.selectbox(
            "Chart type",
            chart_options,
            key="dashboard_chart_type"
        )

        # BAR / LINE
        if chart_type in ["Bar Chart", "Line Chart"] and cat_cols and kpi_cols:

            x = st.selectbox(
                "Category",
                cat_cols,
                key="dashboard_x"
            )

            y = st.selectbox(
                "Numeric value",
                kpi_cols,
                key="dashboard_y"
            )

            fig = make_bar_or_line(
                dashboard_df,
                x,
                y,
                "line" if chart_type == "Line Chart" else "bar",
                f"{y} by {x}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # SCATTER
        elif chart_type == "Scatter Plot" and len(kpi_cols) >= 2:

            x, y = st.columns(2)

            x_col = x.selectbox(
                "X",
                kpi_cols,
                key="dashboard_scatter_x"
            )

            y_col = y.selectbox(
                "Y",
                [c for c in kpi_cols if c != x_col],
                key="dashboard_scatter_y"
            )

            st.plotly_chart(
                px.scatter(
                    dashboard_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                ),
                use_container_width=True
            )

        # HISTOGRAM
        elif chart_type == "Histogram" and kpi_cols:

            col = st.selectbox(
                "Numeric variable",
                kpi_cols,
                key="dashboard_hist"
            )

            st.plotly_chart(
                px.histogram(
                    dashboard_df,
                    x=col,
                    title=f"Distribution of {col}"
                ),
                use_container_width=True
            )

        # PIE
        elif chart_type == "Pie Chart" and cat_cols:

            pie_categories = [
                c for c in cat_cols
                if dashboard_df[c].nunique() <= 20
            ]

            col = st.selectbox(
                "Category",
                pie_categories or cat_cols,
                key="dashboard_pie"
            )

            pie = dashboard_df[col].value_counts().reset_index()
            pie.columns = [col, "Count"]

            st.plotly_chart(
                px.pie(
                    pie,
                    names=col,
                    values="Count",
                    title=f"Distribution of {col}"
                ),
                use_container_width=True
            )

        # BOX PLOT
        elif chart_type == "Box Plot" and kpi_cols:

            col = st.selectbox(
                "Numeric variable",
                kpi_cols,
                key="dashboard_box"
            )

            st.plotly_chart(
                px.box(
                    dashboard_df,
                    y=col,
                    title=f"Box Plot of {col}"
                ),
                use_container_width=True
            )

        else:
            st.info(
                "Not enough suitable columns for this chart type."
            )

if page == "🤖 AI Insights":
    # =========================================================
    # 8. AI INSIGHTS
    # =========================================================

    st.title("🤖 AI Insights")
    st.caption("Let Luffy analyze your dataset and generate actionable insights.")

    insights_language = st.selectbox(
        "🌐 Result language / Langue du résultat",
        ["Français", "English", "العربية", "Español", "Auto (same as AI Chat)"],
        index=0,
        key="ai_insights_language",
        help="Choose the language used for AI Insights. Dataset column names stay unchanged.",
    )

    st.subheader("🧠 AI Analysis")

    st.write(
        "Generate an AI-powered analysis covering data quality, "
        "key insights, correlations, outliers and recommendations."
    )

    if st.button(
        "✨ Generate AI Insights",
        use_container_width=True
    ):
        ctx = dataset_context(working_df)

        corr_text = "No correlations available."

        if len(ctx["numeric"]) >= 2:
            corr = working_df[ctx["numeric"]].corr()

            pairs = []

            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    v = corr.iloc[i, j]

                    if pd.notna(v):
                        pairs.append(
                            (
                                abs(v),
                                corr.columns[i],
                                corr.columns[j],
                                v
                            )
                        )

            pairs.sort(reverse=True)

            if pairs:
                corr_text = "\n".join(
                    f"- {a} / {b}: {v:.2f}"
                    for _, a, b, v in pairs[:5]
                )

        prompt = f"""
You are a professional AI Data Analyst.
LANGUAGE RULE: {language_instruction(insights_language)}
Analyze only the dataset information below. Do not invent facts.

Rows: {ctx['rows']}
Columns: {ctx['columns']}
Numeric columns: {ctx['numeric']}
Categorical columns: {ctx['categorical']}
Missing values: {ctx['missing']}
Duplicates: {ctx['duplicates']}
Correlation highlights:
{corr_text}
Sample:
{ctx['sample']}

Return these sections, translated into the selected language:
1. Dataset overview
2. Data quality
3. Key insights
4. Correlations
5. Potential outliers
6. Recommendations
"""

        try:
            with st.spinner(
                "🤖 Luffy is analyzing your dataset..."
            ):
                answer = ai_chat(prompt)

            st.session_state.ai_insights = answer

        except Exception as e:
            st.error(f"AI error: {e}")

    # =========================================================
    # RESULTS
    # =========================================================

    if st.session_state.get("ai_insights"):

        st.markdown("---")

        st.subheader("📋 Analysis Results")

        st.markdown(
            st.session_state.ai_insights
        )

        st.download_button(
            "⬇️ Download AI Insights",
            st.session_state.ai_insights,
            file_name="dashboard_ai_insights.txt",
            mime="text/plain",
            use_container_width=True
        )

if page == "🧠 AI Agent":
    # =========================================================
    # 9. AI AGENT
    # =========================================================

    st.title("🧠 AI Agent")
    st.caption(
        "Use AI to generate visualizations or run machine learning models "
        "directly on your dataset."
    )

    # =========================================================
    # AI DASHBOARD AGENT
    # =========================================================

    st.subheader("🤖 AI Dashboard Agent")
    st.write(
        "Describe the visualization you want and Luffy will select "
        "the appropriate chart and columns automatically."
    )

    request = st.text_input(
        "What would you like to visualize?",
        placeholder="Example: Show sales by city",
        key="dashboard_agent_request"
    )

    if st.button(
        "🚀 Run Dashboard Agent",
        use_container_width=True
    ):
        if not request.strip():
            st.warning("Please enter a request.")
        else:
            prompt = f"""
You are a data visualization agent.

Columns:
{working_df.columns.tolist()}

Numeric columns:
{num_cols}

Categorical columns:
{cat_cols}

Sample:
{working_df.head(8).to_string(index=False)}

User request:
{request}

Return ONLY valid JSON:

{{
    "chart": "bar|line|scatter|histogram|pie|box",
    "x": null,
    "y": null,
    "title": "..."
}}

Rules:
- Use only existing columns.
- Histogram uses one numeric column.
- Box plot uses one numeric column.
- Scatter uses two numeric columns.
- Bar and line charts require x and y.
- Pie uses one categorical column.
"""

            try:
                with st.spinner("🤖 Luffy is creating your visualization..."):
                    result = safe_json(ai_chat(prompt))

                if not result:
                    st.error("AI did not return valid JSON.")

                else:
                    chart = str(
                        result.get("chart", "")
                    ).lower()

                    x = result.get("x")
                    y = result.get("y")
                    title = result.get(
                        "title",
                        "AI Generated Chart"
                    )

                    valid_charts = {"bar", "line", "scatter", "histogram", "pie", "box"}
                    if chart not in valid_charts:
                        st.error(f"Unsupported chart type: {chart or 'not specified'}")
                    elif x is not None and x not in working_df.columns:
                        st.error(f"Unknown X column: {x}")
                    elif y is not None and y not in working_df.columns:
                        st.error(f"Unknown Y column: {y}")
                    elif chart in {"bar", "line"} and (not x or not y):
                        st.error("Bar and line charts require both X and Y columns.")
                    elif chart in {"scatter"} and (not x or not y):
                        st.error("Scatter plots require both X and Y columns.")
                    elif chart in {"histogram", "pie", "box"} and not x:
                        st.error(f"{chart.title()} charts require an X column.")
                    elif chart in {"bar", "line"} and (x not in working_df.columns or y not in working_df.columns):
                        st.error("Please select valid X and Y columns.")
                    elif chart in {"bar", "line", "scatter"} and (x not in working_df.columns or y not in working_df.columns):
                        st.error("Please select valid X and Y columns.")
                    elif chart in {"histogram", "box"} and x not in num_cols:
                        st.error(f"{chart.title()} charts require a numeric column.")
                    elif chart == "pie" and x not in cat_cols:
                        st.error("Pie charts require a categorical column.")
                    elif chart == "bar" and x and y:
                        st.plotly_chart(
                            make_bar_or_line(
                                working_df,
                                x,
                                y,
                                "bar",
                                title
                            ),
                            use_container_width=True
                        )

                    elif chart == "line" and x and y:
                        st.plotly_chart(
                            make_bar_or_line(
                                working_df,
                                x,
                                y,
                                "line",
                                title
                            ),
                            use_container_width=True
                        )

                    elif chart == "scatter" and x and y and x in num_cols and y in num_cols:
                        st.plotly_chart(
                            px.scatter(
                                working_df,
                                x=x,
                                y=y,
                                title=title
                            ),
                            use_container_width=True
                        )

                    elif chart == "histogram" and x:
                        st.plotly_chart(
                            px.histogram(
                                working_df,
                                x=x,
                                title=title
                            ),
                            use_container_width=True
                        )

                    elif chart == "pie" and x:
                        pie = (
                            working_df[x]
                            .value_counts()
                            .head(20)
                            .reset_index()
                        )

                        pie.columns = [x, "Count"]

                        st.plotly_chart(
                            px.pie(
                                pie,
                                names=x,
                                values="Count",
                                title=title
                            ),
                            use_container_width=True
                        )

                    elif chart == "box" and x:
                        st.plotly_chart(
                            px.box(
                                working_df,
                                y=x,
                                title=title
                            ),
                            use_container_width=True
                        )

                    else:
                        st.warning(
                            "The AI could not create a valid chart "
                            "from the available columns."
                        )

            except Exception as e:
                st.error(
                    f"Dashboard Agent error: {e}"
                )

    # =========================================================
    # MACHINE LEARNING
    # =========================================================

    st.markdown("---")
    st.subheader("🧠 Machine Learning")
    st.caption(
        "Run regression, classification or clustering models "
        "on the current dataset."
    )

    ml_tab1, ml_tab2, ml_tab3 = st.tabs(
        [
            "📈 Regression",
            "🏷️ Classification",
            "🔵 Clustering"
        ]
    )

    # =========================================================
    # REGRESSION
    # =========================================================

    with ml_tab1:

        if len(num_cols) >= 2:

            target = st.selectbox(
                "Target variable",
                num_cols,
                key="reg_target"
            )

            features = st.multiselect(
                "Features",
                [
                    c for c in num_cols
                    if c != target
                ],
                default=[
                    c for c in num_cols
                    if c != target
                ][:2],
                key="reg_features"
            )

            if st.button(
                "🚀 Run Regression",
                key="run_regression",
                use_container_width=True
            ):

                if not features:
                    st.warning(
                        "Select at least one feature."
                    )

                else:

                    data = (
                        working_df[
                            features + [target]
                        ]
                        .dropna()
                    )

                    if len(data) < 10:

                        st.warning(
                            "Not enough complete rows."
                        )

                    else:

                        X_train, X_test, y_train, y_test = (
                            train_test_split(
                                data[features],
                                data[target],
                                test_size=0.2,
                                random_state=42
                            )
                        )

                        model = LinearRegression().fit(
                            X_train,
                            y_train
                        )

                        pred = model.predict(X_test)

                        r2 = r2_score(
                            y_test,
                            pred
                        )

                        rmse = (
                            mean_squared_error(
                                y_test,
                                pred
                            ) ** 0.5
                        )

                        m1, m2 = st.columns(2)

                        m1.metric(
                            "R²",
                            f"{r2:.3f}"
                        )

                        m2.metric(
                            "RMSE",
                            f"{rmse:.3f}"
                        )

                        st.subheader(
                            "📊 Predictions"
                        )

                        st.dataframe(
                            pd.DataFrame(
                                {
                                    "Actual": y_test.values,
                                    "Predicted": pred
                                }
                            ).head(30),
                            use_container_width=True
                        )

        else:

            st.info(
                "Need at least two numeric columns."
            )

        # =========================================================
    # CLASSIFICATION
    # =========================================================

    with ml_tab2:

        if len(num_cols) >= 1 and len(cat_cols) >= 1:

            target = st.selectbox(
                "Target category",
                cat_cols,
                key="clf_target"
            )

            features = st.multiselect(
                "Numeric features",
                num_cols,
                default=num_cols[:2],
                key="clf_features"
            )

            if st.button(
                "🚀 Run Classification",
                key="run_classification",
                use_container_width=True
            ):

                if not features:

                    st.warning(
                        "Select at least one feature."
                    )

                else:

                    data = (
                        working_df[
                            features + [target]
                        ]
                        .dropna()
                        .copy()
                    )

                    if data[target].nunique() < 2:

                        st.warning(
                            "The target needs at least two classes."
                        )

                    elif len(data) < 10:

                        st.warning(
                            "Not enough complete rows."
                        )

                    else:

                        encoder = LabelEncoder()

                        y = encoder.fit_transform(
                            data[target].astype(str)
                        )

                        class_counts = pd.Series(y).value_counts()

                        test_rows = max(
                            1,
                            int(round(len(y) * 0.2))
                        )

                        if (
                            class_counts.min() < 2
                            or test_rows < len(class_counts)
                        ):

                            st.warning(
                                "Not enough samples in each class "
                                "for a stratified train/test split."
                            )

                        else:

                            X_train, X_test, y_train, y_test = (
                                train_test_split(
                                    data[features],
                                    y,
                                    test_size=0.2,
                                    random_state=42,
                                    stratify=y
                                )
                            )

                            model = LogisticRegression(
                                max_iter=1000
                            ).fit(
                                X_train,
                                y_train
                            )

                            pred = model.predict(
                                X_test
                            )

                            accuracy = accuracy_score(
                                y_test,
                                pred
                            )

                            st.metric(
                                "Accuracy",
                                f"{accuracy:.3f}"
                            )

        else:

            st.info(
                "Need numeric features and "
                "a categorical target."
            )


    # =========================================================
    # CLUSTERING
    # =========================================================

    with ml_tab3:

        if len(num_cols) >= 2:

            features = st.multiselect(
                "Clustering features",
                num_cols,
                default=num_cols[:2],
                key="cluster_features"
            )

            k = st.slider(
                "Number of clusters",
                2,
                8,
                3,
                key="cluster_k"
            )

            if st.button(
                "🚀 Run Clustering",
                key="run_clustering",
                use_container_width=True
            ):

                if len(features) < 2:

                    st.warning(
                        "Select at least two features."
                    )

                else:

                    data = (
                        working_df[features]
                        .dropna()
                    )

                    if len(data) < k:

                        st.warning(
                            "Not enough rows for "
                            "the selected number of clusters."
                        )

                    else:

                        scaled = StandardScaler().fit_transform(
                            data
                        )

                        model = KMeans(
                            n_clusters=k,
                            random_state=42,
                            n_init=10
                        )

                        labels = model.fit_predict(
                            scaled
                        )

                        result = data.copy()

                        result["Cluster"] = labels

                        st.subheader(
                            "📋 Cluster Results"
                        )

                        st.dataframe(
                            result.head(100),
                            use_container_width=True
                        )

                        st.plotly_chart(
                            px.scatter(
                                result,
                                x=features[0],
                                y=features[1],
                                color="Cluster",
                                title="Clusters"
                            ),
                            use_container_width=True
                        )

        else:

            st.info(
                "Need at least two numeric columns."
            )
                        
if page == "📄 Reports":
    # 11. FINAL AI REPORT + PDF
    # =========================================================

    st.header("📄 11. AI Report")

    report_language = st.selectbox(
        "🌐 Report language / Langue du rapport",
        ["Français", "English", "العربية", "Español", "Auto (same as AI Chat)"],
        index=0,
        key="ai_report_language",
        help="Choose the language for the TXT and PDF report. Dataset column names stay unchanged.",
    )

    if st.button("Generate Final AI Report", use_container_width=True):
        ctx = dataset_context(working_df)
        prompt = f"""
    Create a concise, professional data analysis report.
    LANGUAGE RULE: {language_instruction(report_language)}
    Use only these facts and the sample. Do not invent.
    Rows: {ctx['rows']}
    Columns: {ctx['columns']}
    Numeric: {ctx['numeric']}
    Categorical: {ctx['categorical']}
    Missing: {ctx['missing']}
    Duplicates: {ctx['duplicates']}
    Sample:
    {ctx['sample']}

    Sections (translate the headings into the selected language):
    1. Executive summary
    2. Data quality
    3. Key insights
    4. Statistical observations
    5. Business recommendations
    """
        try:
            with st.spinner("Generating report..."):
                st.session_state.final_report = ai_chat(prompt)
        except Exception as e:
            st.error(f"Report error: {e}")

    if st.session_state.get("final_report"):
        st.markdown(st.session_state.final_report)
        st.download_button("⬇️ Download TXT Report", st.session_state.final_report,
                           file_name="AI_Data_Analysis_Report.txt", mime="text/plain")
        if st.button("📑 Generate PDF Report", key="generate_pdf"):
            try:
                pdf_bytes = create_pdf(working_df, st.session_state.final_report)
                st.download_button("⬇️ Download PDF Report", pdf_bytes,
                                   file_name="AI_Data_Analysis_Report.pdf", mime="application/pdf", key="download_pdf")
            except Exception as e:
                st.error(f"PDF generation error: {e}")

if page == "💬 AI Chat":
    # 12. AI DATA ANALYST CHAT - SINGLE CONSOLIDATED AGENT
    # =========================================================

    st.header("💬 12. AI Data Analyst Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = {}

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_message = st.chat_input("Ask anything about your dataset...")

    if user_message:
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        ctx = dataset_context(working_df)
        history = st.session_state.chat_history[-10:]
        previous = st.session_state.chat_context

        prompt = f"""
    You are an intelligent AI Data Analyst Agent.
    Dataset rows: {ctx['rows']}
    Dataset columns: {ctx['columns']}
    Numeric columns: {ctx['numeric']}
    Categorical columns: {ctx['categorical']}
    Missing values: {ctx['missing']}
    Duplicates: {ctx['duplicates']}
    Sample:
    {ctx['sample']}
    Previous action context: {previous}
    Recent conversation: {history}

    User request: {user_message}

    You are called Luffy. For casual conversation, greetings, small talk, or questions about your name/identity
    (for example "hi", "how are you?", "what's your name?"), choose action "answer". Do not require dataset context
    for these questions. Reply naturally in the same language as the user's message. Your assistant name is Luffy.
    For dataset questions, answer using the provided dataset context only.
    Choose one action: answer, visualization, regression, classification, clustering, cleaning.
    Return ONLY JSON:
    {{
      "action":"answer|visualization|regression|classification|clustering|cleaning",
      "x":null,
      "y":null,
      "chart":null,
      "target":null,
      "filter_column":null,
      "filter_value":null,
      "operation":null,
      "response":"short explanation"
    }}
    Rules:
    - Never invent columns.
    - Understand references like it, this graph, same data.
    - For visualization use chart bar, line, scatter, histogram, pie or box.
    - For cleaning operation can be remove_duplicates or fill_missing.
    """

        try:
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    action_data = safe_json(ai_chat(prompt))

                if not action_data:
                    answer = "Je n'ai pas compris la demande."
                else:
                    action = action_data.get("action", "answer")
                    x = action_data.get("x")
                    y = action_data.get("y")
                    chart = str(action_data.get("chart") or "").lower()
                    target = action_data.get("target")
                    operation = action_data.get("operation")

                    st.session_state.chat_context = action_data

                    if action == "answer":
                        answer_prompt = f"""
    You are Luffy, a friendly and helpful AI assistant who is also a data analyst.
    First identify the kind of user message:
    - If it is casual conversation, a greeting, small talk, or a question about your name/identity (such as
      "how are you?", "what's your name?", "who are you?"), answer naturally and warmly without forcing a dataset analysis.
      Your name is Luffy. Match the language used by the user.
    - If it is a question about the dataset, answer using only the dataset information below and do not invent numbers.
    - If it is a general non-data question, answer briefly when you can, and be clear if you are unsure.
    Dataset columns: {ctx['columns']}
    Numeric columns: {ctx['numeric']}
    Categorical columns: {ctx['categorical']}
    Dataset sample:
    {ctx['sample']}
    Recent conversation: {history}
    User question: {user_message}
    Be concise, conversational, and helpful.
    """
                        answer = ai_chat(answer_prompt)
                        st.markdown(answer)

                    elif action == "visualization":
                        valid_charts = {"bar", "line", "scatter", "histogram", "pie", "box"}
                        if chart not in valid_charts:
                            answer = "❌ Unsupported chart type. Choose bar, line, scatter, histogram, pie or box."
                        elif not isinstance(x, str) or x not in working_df.columns:
                            answer = f"❌ Column not found: {x}"
                        elif y is not None and (not isinstance(y, str) or y not in working_df.columns):
                            answer = f"❌ Column not found: {y}"
                        elif chart in {"bar", "line", "scatter"} and not y:
                            answer = f"❌ The {chart} chart requires both X and Y columns."
                        elif chart in {"histogram", "box"} and x not in numeric_columns(working_df):
                            answer = f"❌ {chart.title()} requires a numeric column for X."
                        elif chart == "pie" and x not in categorical_columns(working_df):
                            answer = "❌ Pie charts require a categorical column."
                        elif chart == "bar" and y and y in numeric_columns(working_df):
                            st.plotly_chart(make_bar_or_line(working_df, x, y, "bar", f"{y} by {x}"), use_container_width=True)
                            answer = f"📊 Bar chart: {y} by {x}."
                        elif chart == "line" and y and y in numeric_columns(working_df):
                            st.plotly_chart(make_bar_or_line(working_df, x, y, "line", f"{y} by {x}"), use_container_width=True)
                            answer = f"📈 Line chart: {y} by {x}."
                        elif chart == "scatter" and y and x in numeric_columns(working_df) and y in numeric_columns(working_df):
                            st.plotly_chart(px.scatter(working_df, x=x, y=y), use_container_width=True)
                            answer = f"📊 Scatter plot: {y} vs {x}."
                        elif chart == "histogram":
                            st.plotly_chart(px.histogram(working_df, x=x), use_container_width=True)
                            answer = f"📊 Histogram of {x}."
                        elif chart == "pie":
                            pie = working_df[x].value_counts().head(20).reset_index()
                            pie.columns = [x, "Count"]
                            st.plotly_chart(px.pie(pie, names=x, values="Count"), use_container_width=True)
                            answer = f"🥧 Pie chart of {x}."
                        elif chart == "box":
                            st.plotly_chart(px.box(working_df, y=x), use_container_width=True)
                            answer = f"📦 Box plot of {x}."
                        else:
                            answer = "❌ I could not create the requested visualization. Check that the selected columns have the right data types (bar/line need a numeric Y; scatter needs numeric X and Y)."
                        st.markdown(answer)

                    elif action == "cleaning":
                        if operation == "remove_duplicates":
                            before = len(working_df)
                            st.session_state.cleaned_df = working_df.drop_duplicates().copy()
                            removed = before - len(st.session_state.cleaned_df)
                            answer = f"🧹 Removed {removed} duplicate rows."
                        elif operation == "fill_missing":
                            df2 = working_df.copy()
                            for col in numeric_columns(df2):
                                df2[col] = df2[col].fillna(df2[col].median())
                            st.session_state.cleaned_df = df2
                            answer = "🧹 Numeric missing values were filled with the median."
                        else:
                            answer = "❌ Unknown cleaning operation."
                        st.markdown(answer)
                        st.rerun()

                    elif action == "regression":
                        answer = "Use section 10 Machine Learning → Regression for a controlled regression run."
                        st.markdown(answer)
                    elif action == "classification":
                        answer = "Use section 10 Machine Learning → Classification for a controlled classification run."
                        st.markdown(answer)
                    elif action == "clustering":
                        answer = "Use section 10 Machine Learning → Clustering for a controlled clustering run."
                        st.markdown(answer)
                    else:
                        answer = "❌ Unsupported action."
                        st.markdown(answer)

                st.session_state.chat_history.append({"role": "assistant", "content": answer})

        except Exception as e:
            answer = f"❌ Error: {e}"
            st.error(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})


# FOOTER
# =========================================================

st.markdown("---")
st.caption("AI Data Analyst • CSV/Excel • EDA • Dashboard • ML • AI Insights • Reports")