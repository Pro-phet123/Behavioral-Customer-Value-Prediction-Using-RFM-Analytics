# ============================================================
# ◈ CUSTOMER VALUE INTELLIGENCE
# Premium Responsive Streamlit RFM ML Dashboard
#
# Author:
# Olalemi Olaoluwakintan Emmanuel
#
# Project:
# Behavioral Customer Value Prediction Using RFM Analytics
#
# Model:
# Logistic Regression + SMOTE
# Time-aware / Leakage-aware modelling
#
# IMPORTANT:
# This application is a portfolio/research prototype.
# Predictions are model-based estimates and are not guarantees
# of future customer spending or behaviour.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Value Intelligence",
    page_icon="⬢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "customer_value_pipeline.pkl"


# ============================================================
# EXPECTED MODEL FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "Recency",
    "Frequency",
    "Average_Order_Value",
    "Average_Quantity",
    "Total_Items",
    "Unique_Products",
    "Number_of_Transactions"
]


# ============================================================
# PROJECT INFORMATION
# ============================================================

RAW_TRANSACTIONS = 541_909
CUSTOMER_RECORDS = 3_616
BEHAVIOURAL_FEATURES = 7
FUTURE_SPEND_MEDIAN = 439.61


# ============================================================
# MODEL PERFORMANCE
# ============================================================

BASELINE_METRICS = {

    "Accuracy": 0.7873,
    "Precision": 0.7297,
    "Recall": 0.3951,
    "F1 Score": 0.5127
}


SMOTE_METRICS = {

    "Accuracy": 0.7652,
    "Precision": 0.5792,
    "Recall": 0.6244,
    "F1 Score": 0.6009
}


# ============================================================
# BEST MODEL
# ============================================================

BEST_MODEL = {

    "Model": "Logistic Regression + SMOTE",
    "Validation": "Time-aware",
    "Scaling": "StandardScaler",
    "Balancing": "SMOTE",
    "Selection Metric": "F1 Score"
}


# ============================================================
# SESSION STATE
# ============================================================

if "theme" not in st.session_state:

    st.session_state.theme = "Dark"


if "prediction_result" not in st.session_state:

    st.session_state.prediction_result = None


# ============================================================
# THEME DEFINITIONS
# ============================================================

THEMES = {

    "Dark": {

        "background": "#050505",
        "surface": "#0b0b0d",
        "surface2": "#111114",
        "card": "#111114",

        "border": "rgba(255,255,255,0.09)",

        "text": "#f8fafc",
        "muted": "#94a3b8",

        "primary": "#818cf8",
        "secondary": "#38bdf8",

        "success": "#34d399",
        "warning": "#fbbf24",
        "danger": "#fb7185"
    },


    "Midnight Blue": {

        "background": "#020617",
        "surface": "#0f172a",
        "surface2": "#172033",
        "card": "#0f172a",

        "border": "rgba(148,163,184,0.12)",

        "text": "#f8fafc",
        "muted": "#94a3b8",

        "primary": "#38bdf8",
        "secondary": "#818cf8",

        "success": "#34d399",
        "warning": "#fbbf24",
        "danger": "#fb7185"
    },


    "Emerald": {

        "background": "#020807",
        "surface": "#071311",
        "surface2": "#0c1d19",
        "card": "#071311",

        "border": "rgba(52,211,153,0.12)",

        "text": "#ecfdf5",
        "muted": "#94a3b8",

        "primary": "#34d399",
        "secondary": "#2dd4bf",

        "success": "#4ade80",
        "warning": "#fbbf24",
        "danger": "#fb7185"
    },


    "Light": {

        "background": "#f5f7fb",
        "surface": "#ffffff",
        "surface2": "#f8fafc",
        "card": "#ffffff",

        "border": "rgba(15,23,42,0.08)",

        "text": "#0f172a",
        "muted": "#64748b",

        "primary": "#4f46e5",
        "secondary": "#0284c7",

        "success": "#16a34a",
        "warning": "#d97706",
        "danger": "#dc2626"
    }
}


theme = THEMES[st.session_state.theme]


# ============================================================
# GLOBAL CSS
# ============================================================

st.html(
    f"""
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    * {{
        box-sizing: border-box;
    }}

    html,
    body {{
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }}

    .stApp {{

        background:

            radial-gradient(
                circle at 10% 0%,
                rgba(99,102,241,0.10),
                transparent 28%
            ),

            radial-gradient(
                circle at 90% 10%,
                rgba(14,165,233,0.08),
                transparent 25%
            ),

            {theme["background"]};

        color: {theme["text"]};
    }}


    /* ======================================================
       MAIN CONTAINER
       ====================================================== */

    .block-container {{

        max-width: 1450px;

        padding-top: 2rem;
        padding-bottom: 5rem;

        padding-left: 3rem;
        padding-right: 3rem;
    }}


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {{

        background: {theme["surface"]};

        border-right:
            1px solid {theme["border"]};
    }}

    section[data-testid="stSidebar"] * {{
        color: {theme["text"]};
    }}


    /* ======================================================
       HEADINGS
       ====================================================== */

    h1,
    h2,
    h3,
    h4,
    p {{
        color: {theme["text"]};
    }}


    /* ======================================================
       HERO
       ====================================================== */

    .hero {{

        position: relative;

        overflow: hidden;

        padding: 3rem;

        border-radius: 30px;

        margin-bottom: 1.5rem;

        background:

            radial-gradient(
                circle at 85% 15%,
                rgba(129,140,248,0.22),
                transparent 30%
            ),

            linear-gradient(
                135deg,
                {theme["surface"]},
                {theme["surface2"]}
            );

        border:
            1px solid {theme["border"]};

        box-shadow:
            0 25px 80px rgba(0,0,0,0.25);
    }}


    .hero-badge {{

        display: inline-block;

        padding: 7px 13px;

        border-radius: 999px;

        background:
            rgba(129,140,248,0.10);

        border:
            1px solid rgba(129,140,248,0.22);

        color:
            {theme["primary"]};

        font-size: 0.75rem;

        font-weight: 700;

        letter-spacing: 0.08em;

        margin-bottom: 1rem;
    }}


    .hero-title {{

        margin: 0;

        font-size:
            clamp(2.4rem, 6vw, 5rem);

        line-height: 0.98;

        letter-spacing: -0.055em;

        font-weight: 850;

        color:
            {theme["text"]};
    }}


    .hero-title span {{
        color: {theme["primary"]};
    }}


    .hero-subtitle {{

        max-width: 800px;

        margin-top: 1.3rem;

        color:
            {theme["muted"]};

        font-size: 1.05rem;

        line-height: 1.7;
    }}


    /* ======================================================
       STATUS
       ====================================================== */

    .status {{

        display: inline-block;

        padding: 7px 12px;

        border-radius: 999px;

        font-size: 0.78rem;

        font-weight: 700;
    }}


    .status-green {{

        color:
            {theme["success"]};

        background:
            rgba(52,211,153,0.09);

        border:
            1px solid rgba(52,211,153,0.18);
    }}


    .status-blue {{

        color:
            {theme["primary"]};

        background:
            rgba(129,140,248,0.09);

        border:
            1px solid rgba(129,140,248,0.18);
    }}


    /* ======================================================
       SAFETY / MODEL NOTICE
       ====================================================== */

    .notice {{

        padding:
            1.2rem 1.4rem;

        border-radius: 18px;

        margin-bottom: 1.6rem;

        background:
            rgba(129,140,248,0.06);

        border:
            1px solid rgba(129,140,248,0.18);

        color:
            {theme["text"]};

        line-height: 1.65;
    }}


    .notice-title {{

        font-weight: 800;

        color:
            {theme["primary"]};

        margin-bottom: 0.35rem;
    }}


    /* ======================================================
       SECTION
       ====================================================== */

    .section-title {{

        font-size: 1.8rem;

        font-weight: 800;

        letter-spacing: -0.035em;

        margin:
            1.5rem 0 0.4rem 0;
    }}


    .section-description {{

        color:
            {theme["muted"]};

        margin-bottom: 1.2rem;

        line-height: 1.6;
    }}


    /* ======================================================
       METRIC CARDS
       ====================================================== */

    .metric-card {{

        padding: 1.25rem;

        min-height: 135px;

        border-radius: 20px;

        background:
            {theme["card"]};

        border:
            1px solid {theme["border"]};

        box-shadow:
            0 10px 35px rgba(0,0,0,0.12);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease;
    }}


    .metric-card:hover {{

        transform:
            translateY(-3px);

        border-color:
            rgba(129,140,248,0.30);
    }}


    .metric-label {{

        color:
            {theme["muted"]};

        font-size: 0.72rem;

        font-weight: 700;

        letter-spacing: 0.07em;

        text-transform: uppercase;
    }}


    .metric-value {{

        color:
            {theme["text"]};

        font-size: 2rem;

        font-weight: 850;

        margin-top: 0.3rem;

        letter-spacing: -0.04em;
    }}


    .metric-delta {{

        color:
            {theme["success"]};

        font-size: 0.75rem;

        margin-top: 0.3rem;
    }}


    /* ======================================================
       GENERAL CARDS
       ====================================================== */

    .card {{

        padding: 1.5rem;

        border-radius: 22px;

        background:
            {theme["card"]};

        border:
            1px solid {theme["border"]};

        margin-bottom: 1rem;

        box-shadow:
            0 10px 40px rgba(0,0,0,0.10);
    }}


    .card h3 {{

        margin-top: 0;

        color:
            {theme["text"]};
    }}


    .card p {{

        color:
            {theme["muted"]};

        line-height: 1.7;
    }}


    /* ======================================================
       FEATURE ROW
       ====================================================== */

    .feature-row {{

        display: flex;

        justify-content:
            space-between;

        align-items:
            center;

        padding:
            0.75rem 0;

        border-bottom:
            1px solid {theme["border"]};
    }}


    .feature-row:last-child {{
        border-bottom: none;
    }}


    .feature-name {{

        color:
            {theme["text"]};

        font-weight: 600;
    }}


    .feature-value {{

        color:
            {theme["primary"]};

        font-weight: 800;

        font-family:
            monospace;
    }}


    /* ======================================================
       RESULT CARDS
       ====================================================== */

    .result-high {{

        padding: 1.6rem;

        border-radius: 24px;

        background:
            rgba(52,211,153,0.08);

        border:
            1px solid rgba(52,211,153,0.28);

        margin-top: 1rem;
    }}


    .result-low {{

        padding: 1.6rem;

        border-radius: 24px;

        background:
            rgba(148,163,184,0.08);

        border:
            1px solid rgba(148,163,184,0.22);

        margin-top: 1rem;
    }}


    .result-title {{

        font-size: 1.35rem;

        font-weight: 800;

        margin-bottom: 0.5rem;
    }}


    .result-probability {{

        font-size:
            clamp(2.4rem, 5vw, 4rem);

        font-weight: 850;

        letter-spacing: -0.05em;

        color:
            {theme["primary"]};

        margin-top: 0.7rem;
    }}


    .result-text {{

        color:
            {theme["muted"]};

        line-height: 1.65;
    }}


    /* ======================================================
       RFM BADGES
       ====================================================== */

    .rfm-badge {{

        display: inline-block;

        padding: 6px 10px;

        margin-right: 5px;

        border-radius: 999px;

        color:
            {theme["primary"]};

        background:
            rgba(129,140,248,0.09);

        border:
            1px solid rgba(129,140,248,0.20);

        font-size: 0.72rem;

        font-weight: 800;
    }}


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {{

        text-align: center;

        margin-top: 4rem;

        padding: 2rem;

        color:
            {theme["muted"]};

        border-top:
            1px solid {theme["border"]};
    }}


    /* ======================================================
       MOBILE RESPONSIVENESS
       ====================================================== */

    @media (max-width: 768px) {{

        .block-container {{

            padding-left: 1rem;

            padding-right: 1rem;

            padding-top: 1rem;
        }}


        .hero {{

            padding: 1.7rem;

            border-radius: 22px;
        }}


        .hero-title {{

            font-size:
                2.7rem;
        }}


        .hero-subtitle {{

            font-size:
                0.95rem;
        }}


        .metric-card {{

            min-height:
                110px;
        }}


        .metric-value {{

            font-size:
                1.55rem;
        }}


        .result-probability {{

            font-size:
                2.7rem;
        }}

    }}

    </style>
    """
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        return None, (
            f"Model file not found: "
            f"{MODEL_PATH.name}"
        )

    try:

        loaded_model = joblib.load(
            MODEL_PATH
        )

        return loaded_model, None

    except Exception as error:

        return None, str(error)


model, model_error = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ◈ CVI"
    )

    st.caption(
        "Customer Value Intelligence"
    )

    st.divider()


    # --------------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------------

    page = st.radio(

        "Navigation",

        [
            "🏠 Overview",
            "🔮 Prediction",
            "👤 Customer Profile",
            "🧠 Model Context",
            "⚖️ Evaluation",
            "📘 Methodology"
        ],

        label_visibility="collapsed"
    )


    st.divider()


    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    st.markdown(
        "### Model"
    )


    if model is not None:

        st.markdown(
            """
            <span class="rfm-badge">
                Logistic Regression
            </span>

            <span class="rfm-badge">
                SMOTE
            </span>

            <span class="rfm-badge">
                Time-aware
            </span>
            """,
            unsafe_allow_html=True
        )

        st.success(
            "Model loaded"
        )

    else:

        st.error(
            "Model unavailable"
        )


    st.divider()


    # --------------------------------------------------------
    # THEME
    # --------------------------------------------------------

    st.markdown(
        "### ⚙️ Appearance"
    )


    selected_theme = st.selectbox(

        "Theme",

        list(THEMES.keys()),

        index=list(
            THEMES.keys()
        ).index(
            st.session_state.theme
        ),

        label_visibility="collapsed"
    )


    if selected_theme != st.session_state.theme:

        st.session_state.theme = (
            selected_theme
        )

        st.rerun()


    st.divider()


    st.caption(
        "Behavioral Customer Value Prediction"
    )

    st.caption(
        "RFM + behavioural analytics"
    )


# ============================================================
# SYSTEM BAR
# ============================================================

st.html(
    f"""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        margin-bottom:1rem;
        gap:1rem;
        flex-wrap:wrap;
    ">

        <span class="status status-green">
            ● SYSTEM ONLINE
        </span>

        <span style="
            color:{theme["muted"]};
            font-size:0.8rem;
        ">
            Customer Intelligence • Predictive Analytics
        </span>

    </div>
    """
)


# ============================================================
# HERO
# ============================================================

st.html(
    f"""
    <div class="hero">

        <div class="hero-badge">
            ◈ CUSTOMER INTELLIGENCE • PREDICTIVE ANALYTICS
        </div>

        <h1 class="hero-title">

            Customer  
            <br>

            <span>Value</span>
            <br>

            Intelligence

        </h1>

        <p class="hero-subtitle">

            Turn historical purchasing behaviour into an
            actionable customer-value signal. Explore RFM
            and behavioural features and estimate the likelihood
            that a customer belongs to the high-value segment.

        </p>

        <br>

        <span class="status status-green">
            ● Leakage-aware modelling
        </span>

    </div>
    """
)


# ============================================================
# PROJECT NOTICE
# ============================================================

st.html(
    f"""
    <div class="notice">

        <div class="notice-title">
            ◈ About this prediction system
        </div>

        This application is a portfolio and research prototype
        demonstrating customer-value prediction using historical
        purchasing behaviour. Model probabilities are estimates
        generated by the trained machine-learning pipeline and
        should not be interpreted as guarantees of future revenue
        or customer behaviour.

    </div>
    """
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":


    st.html(
        """
        <div class="section-title">
            Customer Value Intelligence
        </div>

        <div class="section-description">
            A predictive analytics application designed to
            transform historical transaction behaviour into
            customer-level value signals.
        </div>
        """
    )


    # ========================================================
    # TOP METRICS
    # ========================================================

    m1, m2, m3, m4 = st.columns(4)


    overview_metrics = [

        (
            m1,
            "RAW TRANSACTIONS",
            f"{RAW_TRANSACTIONS:,}",
            "Original retail transactions"
        ),

        (
            m2,
            "CUSTOMER RECORDS",
            f"{CUSTOMER_RECORDS:,}",
            "Customer-level modelling data"
        ),

        (
            m3,
            "BEHAVIOURAL FEATURES",
            str(BEHAVIOURAL_FEATURES),
            "Features used for prediction"
        ),

        (
            m4,
            "FUTURE-SPEND MEDIAN",
            f"£{FUTURE_SPEND_MEDIAN:,.2f}",
            "High-value target threshold"
        )
    ]


    for col, label, value, delta in overview_metrics:

        with col:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="metric-value">
                        {value}
                    </div>

                    <div class="metric-delta">
                        {delta}
                    </div>

                </div>
                """
            )


    # ========================================================
    # WHAT IS RFM?
    # ========================================================

    st.html(
        """
        <div class="section-title">
            What is behind the prediction?
        </div>

        <div class="section-description">
            The model uses historical customer behaviour to
            identify patterns associated with future high-value
            spending.
        </div>
        """
    )


    r1, r2, r3 = st.columns(3)


    with r1:

        st.html(
            f"""
            <div class="card">

                <h3>R — Recency</h3>

                <p>
                    How recently did the customer purchase?
                    Lower recency generally represents more
                    recent purchasing activity.
                </p>

                <span class="rfm-badge">
                    Behavioural signal
                </span>

            </div>
            """
        )


    with r2:

        st.html(
            f"""
            <div class="card">

                <h3>F — Frequency</h3>

                <p>
                    How often does the customer purchase?
                    Higher frequency can indicate stronger
                    engagement with the business.
                </p>

                <span class="rfm-badge">
                    Engagement signal
                </span>

            </div>
            """
        )


    with r3:

        st.html(
            f"""
            <div class="card">

                <h3>M — Monetary Behaviour</h3>

                <p>
                    How much value does the customer generate
                    through purchasing behaviour and order
                    characteristics?
                </p>

                <span class="rfm-badge">
                    Value signal
                </span>

            </div>
            """
        )


    # ========================================================
    # MODELLING LOGIC
    # ========================================================

    st.html(
        """
        <div class="section-title">
            Past behaviour → prediction → future outcome
        </div>
        """
    )


    st.html(
        f"""
        <div class="card">

            <p>

                Historical transactions before
                <strong>2011-10-01</strong>
                were used to engineer customer behaviour.

            </p>

            <p>

                Future transactions from
                <strong>2011-10-01</strong>
                onward were used to calculate future spending
                and define the high-value customer target.

            </p>

            <p>

                This time-aware structure was designed to reduce
                target leakage by ensuring that future information
                was not used to construct historical behavioural
                features.

            </p>

        </div>
        """
    )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.html(
        """
        <div class="section-title">
            Model Performance
        </div>
        """
    )


    p1, p2, p3, p4 = st.columns(4)


    performance_cards = [

        (
            p1,
            "ACCURACY",
            f"{SMOTE_METRICS['Accuracy']:.2%}",
            "Logistic + SMOTE"
        ),

        (
            p2,
            "PRECISION",
            f"{SMOTE_METRICS['Precision']:.2%}",
            "Logistic + SMOTE"
        ),

        (
            p3,
            "RECALL",
            f"{SMOTE_METRICS['Recall']:.2%}",
            "Logistic + SMOTE"
        ),

        (
            p4,
            "F1 SCORE",
            f"{SMOTE_METRICS['F1 Score']:.2%}",
            "Best selected model"
        )
    ]


    for col, label, value, delta in performance_cards:

        with col:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="metric-value">
                        {value}
                    </div>

                    <div class="metric-delta">
                        {delta}
                    </div>

                </div>
                """
            )


# ============================================================
# PAGE 2 — PREDICTION
# ============================================================

elif page == "🔮 Prediction":


    st.html(
        """
        <div class="section-title">
            🔮 Customer Value Prediction
        </div>

        <div class="section-description">
            Enter a customer's historical behavioural profile
            to estimate the probability of belonging to the
            high-value segment.
        </div>
        """
    )


    # ========================================================
    # MODEL CHECK
    # ========================================================

    if model is None:

        st.error(
            "The customer-value model could not be loaded."
        )

        if model_error:

            st.code(
                model_error
            )

        st.stop()


    # ========================================================
    # MODEL CARD
    # ========================================================

    st.html(
        f"""
        <div class="card">

            <h3>Active Prediction Model</h3>

            <p>

                <strong>Logistic Regression + SMOTE</strong>

                <br>

                The deployed pipeline contains the fitted
                preprocessing and trained Logistic Regression
                model.

            </p>

            <span class="rfm-badge">
                StandardScaler
            </span>

            <span class="rfm-badge">
                Logistic Regression
            </span>

            <span class="rfm-badge">
                SMOTE
            </span>

            <span class="rfm-badge">
                Time-aware
            </span>

        </div>
        """
    )


    # ========================================================
    # INPUT FORM
    # ========================================================

    with st.form(
        "customer_prediction_form"
    ):

        st.markdown(
            "### Customer Behavioural Profile"
        )


        c1, c2, c3 = st.columns(3)


        # ----------------------------------------------------
        # COLUMN 1
        # ----------------------------------------------------

        with c1:

            recency = st.number_input(

                "Recency (days)",

                min_value=0.0,

                max_value=5000.0,

                value=30.0,

                step=1.0,

                help=(
                    "Number of days since the customer's "
                    "most recent purchase."
                )
            )


            frequency = st.number_input(

                "Frequency (orders)",

                min_value=1.0,

                max_value=5000.0,

                value=3.0,

                step=1.0,

                help=(
                    "Number of customer transactions/orders."
                )
            )


            average_order_value = st.number_input(

                "Average Order Value",

                min_value=0.0,

                max_value=100000.0,

                value=150.0,

                step=10.0,

                help=(
                    "Average monetary value of each order."
                )
            )


        # ----------------------------------------------------
        # COLUMN 2
        # ----------------------------------------------------

        with c2:

            average_quantity = st.number_input(

                "Average Quantity",

                min_value=0.0,

                max_value=10000.0,

                value=10.0,

                step=1.0,

                help=(
                    "Average number of items per transaction."
                )
            )


            total_items = st.number_input(

                "Total Items",

                min_value=0.0,

                max_value=100000.0,

                value=30.0,

                step=1.0,

                help=(
                    "Total number of items purchased."
                )
            )


            unique_products = st.number_input(

                "Unique Products",

                min_value=0.0,

                max_value=10000.0,

                value=15.0,

                step=1.0,

                help=(
                    "Number of distinct products purchased."
                )
            )


        # ----------------------------------------------------
        # COLUMN 3
        # ----------------------------------------------------

        with c3:

            number_of_transactions = st.number_input(

                "Number of Transactions",

                min_value=1.0,

                max_value=5000.0,

                value=5.0,

                step=1.0,

                help=(
                    "Total number of transactions."
                )
            )


            st.markdown(
                "#### Behavioural interpretation"
            )


            st.caption(
                "Lower recency generally means more recent "
                "activity. Higher frequency, order value, "
                "product diversity and transaction volume "
                "may indicate stronger historical customer "
                "engagement."
            )


        submitted = st.form_submit_button(

            "🔮 RUN CUSTOMER VALUE PREDICTION",

            use_container_width=True

        )


    # ========================================================
    # PREDICT
    # ========================================================

    if submitted:


        input_data = pd.DataFrame(

            [[

                recency,

                frequency,

                average_order_value,

                average_quantity,

                total_items,

                unique_products,

                number_of_transactions

            ]],

            columns=EXPECTED_FEATURES

        )


        try:


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = int(

                model.predict(
                    input_data
                )[0]

            )


            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probability = float(

                    model.predict_proba(
                        input_data
                    )[0, 1]

                )

            else:

                probability = float(
                    prediction
                )


            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

            st.session_state.prediction_result = {

                "prediction": prediction,

                "probability": probability,

                "input_data": input_data

            }


        except Exception as error:

            st.session_state.prediction_result = None

            st.error(
                "Prediction failed."
            )

            st.exception(
                error
            )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    result = (
        st.session_state.prediction_result
    )


    if result is not None:


        prediction = result[
            "prediction"
        ]

        probability = result[
            "probability"
        ]

        input_data = result[
            "input_data"
        ]


        st.divider()


        st.html(
            """
            <div class="section-title">
                Prediction Result
            </div>
            """
        )


        probability_percent = (
            probability * 100
        )


        # ====================================================
        # RESULT
        # ====================================================

        if prediction == 1:

            st.html(
                f"""
                <div class="result-high">

                    <div class="result-title">
                        ✓ High-value customer signal detected
                    </div>

                    <div class="result-text">

                        The customer's behavioural profile
                        is classified by the model as belonging
                        to the high-value segment.

                    </div>

                    <div class="result-probability">
                        {probability_percent:.1f}%
                    </div>

                    <div class="result-text">

                        Estimated probability of the
                        high-value class.

                    </div>

                </div>
                """
            )


        else:

            st.html(
                f"""
                <div class="result-low">

                    <div class="result-title">
                        Lower high-value probability
                    </div>

                    <div class="result-text">

                        The customer's current behavioural
                        profile is less aligned with the
                        high-value segment according to
                        the trained model.

                    </div>

                    <div class="result-probability">
                        {probability_percent:.1f}%
                    </div>

                    <div class="result-text">

                        Estimated probability of the
                        high-value class.

                    </div>

                </div>
                """
            )


        # ====================================================
        # PROBABILITY PROGRESS
        # ====================================================

        st.markdown(
            "### High-value probability"
        )


        st.progress(
            min(
                max(
                    probability,
                    0.0
                ),
                1.0
            )
        )


        # ====================================================
        # RESULT METRICS
        # ====================================================

        q1, q2, q3 = st.columns(3)


        with q1:

            st.metric(
                "Probability",
                f"{probability_percent:.1f}%"
            )


        with q2:

            st.metric(
                "Classification",
                (
                    "High Value"
                    if prediction == 1
                    else "Lower Value"
                )
            )


        with q3:

            st.metric(
                "Model",
                "Logistic + SMOTE"
            )


        # ====================================================
        # BEHAVIOURAL SNAPSHOT
        # ====================================================

        st.markdown(
            "### Behavioural Snapshot"
        )


        profile = pd.DataFrame(

            {

                "Feature":
                    EXPECTED_FEATURES,

                "Customer Value":
                    input_data.iloc[0].values

            }

        )


        st.dataframe(

            profile,

            use_container_width=True,

            hide_index=True

        )


        st.caption(

            "The probability is a model-based estimate and "
            "should not be interpreted as a guarantee of "
            "future customer spending."

        )


# ============================================================
# PAGE 3 — CUSTOMER PROFILE
# ============================================================

elif page == "👤 Customer Profile":


    st.html(
        """
        <div class="section-title">
            👤 Customer Profile Simulator
        </div>

        <div class="section-description">
            Explore how changes in behavioural characteristics
            change the customer profile before running a formal
            prediction.
        </div>
        """
    )


    # ========================================================
    # DEFAULT PROFILE
    # ========================================================

    default_profile = {

        "Recency": 30.0,

        "Frequency": 3.0,

        "Average_Order_Value": 150.0,

        "Average_Quantity": 10.0,

        "Total_Items": 30.0,

        "Unique_Products": 15.0,

        "Number_of_Transactions": 5.0

    }


    # ========================================================
    # SLIDERS
    # ========================================================

    values = {}


    s1, s2 = st.columns(2)


    for index, feature in enumerate(
        EXPECTED_FEATURES
    ):


        container = (
            s1
            if index % 2 == 0
            else s2
        )


        with container:


            base_value = (
                default_profile[
                    feature
                ]
            )


            maximum = max(
                base_value * 10,
                100
            )


            values[feature] = st.slider(

                feature.replace(
                    "_",
                    " "
                ),

                min_value=0.0,

                max_value=float(
                    maximum
                ),

                value=float(
                    base_value
                )

            )


    # ========================================================
    # PROFILE DATAFRAME
    # ========================================================

    profile = pd.DataFrame(
        [values]
    )


    st.markdown(
        "### Current Behavioural Profile"
    )


    st.dataframe(

        profile,

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # PROFILE SUMMARY
    # ========================================================

    st.html(
        f"""
        <div class="card">

            <h3>Behavioural Summary</h3>

            <div class="feature-row">

                <span class="feature-name">
                    Recency
                </span>

                <span class="feature-value">
                    {values["Recency"]:.1f}
                </span>

            </div>

            <div class="feature-row">

                <span class="feature-name">
                    Frequency
                </span>

                <span class="feature-value">
                    {values["Frequency"]:.1f}
                </span>

            </div>

            <div class="feature-row">

                <span class="feature-name">
                    Average Order Value
                </span>

                <span class="feature-value">
                    {values["Average_Order_Value"]:.1f}
                </span>

            </div>

            <div class="feature-row">

                <span class="feature-name">
                    Unique Products
                </span>

                <span class="feature-value">
                    {values["Unique_Products"]:.1f}
                </span>

            </div>

        </div>
        """
    )


    st.info(

        "This simulator is intended for behavioural "
        "exploration. The Prediction page uses the saved "
        "machine-learning pipeline."

    )


# ============================================================
# PAGE 4 — MODEL CONTEXT
# ============================================================

elif page == "🧠 Model Context":


    st.html(
        """
        <div class="section-title">
            🧠 Model Context
        </div>

        <div class="section-description">
            Understand the dataset construction, behavioural
            features and modelling logic behind the deployed
            customer-value prediction system.
        </div>
        """
    )


    # ========================================================
    # DATASET METRICS
    # ========================================================

    m1, m2, m3, m4 = st.columns(4)


    context_metrics = [

        (
            m1,
            "RAW TRANSACTIONS",
            f"{RAW_TRANSACTIONS:,}",
            "Online retail dataset"
        ),

        (
            m2,
            "CUSTOMER RECORDS",
            f"{CUSTOMER_RECORDS:,}",
            "Customer-level aggregation"
        ),

        (
            m3,
            "FEATURES",
            str(BEHAVIOURAL_FEATURES),
            "Behavioural variables"
        ),

        (
            m4,
            "FUTURE-SPEND MEDIAN",
            f"£{FUTURE_SPEND_MEDIAN:,.2f}",
            "High-value threshold"
        )

    ]


    for col, label, value, delta in context_metrics:

        with col:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="metric-value">
                        {value}
                    </div>

                    <div class="metric-delta">
                        {delta}
                    </div>

                </div>
                """
            )


    # ========================================================
    # LEAKAGE-AWARE DESIGN
    # ========================================================

    st.html(
        """
        <div class="section-title">
            The Modelling Logic
        </div>
        """
    )


    st.html(
        """
        <div class="card">

            <h3>
                Past behaviour → prediction → future outcome
            </h3>

            <p>

                Historical transactions before
                <strong>2011-10-01</strong>
                were used to engineer customer-level
                behavioural features.

            </p>

            <p>

                Future transactions from
                <strong>2011-10-01</strong>
                onward were used to calculate future spending
                and define the high-value target.

            </p>

            <p>

                The purpose of this temporal separation was to
                avoid allowing future purchasing information to
                leak into the behavioural features used for
                prediction.

            </p>

        </div>
        """
    )


    # ========================================================
    # SEVEN FEATURES
    # ========================================================

    st.markdown(
        "### Seven Behavioural Features"
    )


    feature_descriptions = {

        "Recency":
            "Days since the customer's most recent purchase.",

        "Frequency":
            "Number of customer orders/purchases.",

        "Average_Order_Value":
            "Average monetary value per order.",

        "Average_Quantity":
            "Average quantity of items per transaction.",

        "Total_Items":
            "Total number of items purchased.",

        "Unique_Products":
            "Number of distinct products purchased.",

        "Number_of_Transactions":
            "Total transaction count used in the behavioural profile."
    }


    feature_table = pd.DataFrame(

        {

            "Feature":
                list(
                    feature_descriptions.keys()
                ),

            "Description":
                list(
                    feature_descriptions.values()
                )

        }

    )


    st.dataframe(

        feature_table,

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # MODELS COMPARED
    # ========================================================

    st.markdown(
        "### Models Compared"
    )


    models_compared = pd.DataFrame(

        {

            "Model": [

                "Logistic Regression",

                "Decision Tree",

                "Random Forest",

                "XGBoost"

            ],

            "SMOTE": [

                "Applied",

                "Applied",

                "Applied",

                "Applied"

            ],

            "Purpose": [

                "Linear baseline",

                "Non-linear tree model",

                "Ensemble model",

                "Gradient boosting model"

            ]

        }

    )


    st.dataframe(

        models_compared,

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # DEPLOYMENT PIPELINE
    # ========================================================

    st.html(
        """
        <div class="section-title">
            Deployment Pipeline
        </div>

        <div class="card">

            <h3>
                Single sklearn Pipeline
            </h3>

            <p>

                Customer Input

                →
                Seven Behavioural Features

                →
                StandardScaler

                →
                Logistic Regression

                →
                High-value Probability

            </p>

            <p>

                The deployed application uses the saved
                <strong>customer_value_pipeline.pkl</strong>
                model pipeline so that preprocessing and
                prediction remain consistent with the trained
                deployment model.

            </p>

        </div>
        """
    )


# ============================================================
# PAGE 5 — EVALUATION
# ============================================================

elif page == "⚖️ Evaluation":


    st.html(
        """
        <div class="section-title">
            ⚖️ Model Evaluation
        </div>

        <div class="section-description">
            Comparison of the baseline Logistic Regression model
            with the Logistic Regression model trained with SMOTE.
        </div>
        """
    )


    # ========================================================
    # METRIC CARDS
    # ========================================================

    e1, e2, e3, e4 = st.columns(4)


    evaluation_cards = [

        (
            e1,
            "ACCURACY",
            f"{SMOTE_METRICS['Accuracy']:.2%}",
            "Logistic + SMOTE"
        ),

        (
            e2,
            "PRECISION",
            f"{SMOTE_METRICS['Precision']:.2%}",
            "Logistic + SMOTE"
        ),

        (
            e3,
            "RECALL",
            f"{SMOTE_METRICS['Recall']:.2%}",
            "Logistic + SMOTE"
        ),

        (
            e4,
            "F1 SCORE",
            f"{SMOTE_METRICS['F1 Score']:.2%}",
            "Best by F1"
        )

    ]


    for col, label, value, delta in evaluation_cards:

        with col:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="metric-value">
                        {value}
                    </div>

                    <div class="metric-delta">
                        {delta}
                    </div>

                </div>
                """
            )


    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    st.markdown(
        "### Baseline vs Logistic + SMOTE"
    )


    comparison = pd.DataFrame(

        {

            "Metric": [

                "Accuracy",

                "Precision",

                "Recall",

                "F1 Score"

            ],

            "Baseline Logistic": [

                BASELINE_METRICS[
                    "Accuracy"
                ],

                BASELINE_METRICS[
                    "Precision"
                ],

                BASELINE_METRICS[
                    "Recall"
                ],

                BASELINE_METRICS[
                    "F1 Score"
                ]

            ],

            "Logistic + SMOTE": [

                SMOTE_METRICS[
                    "Accuracy"
                ],

                SMOTE_METRICS[
                    "Precision"
                ],

                SMOTE_METRICS[
                    "Recall"
                ],

                SMOTE_METRICS[
                    "F1 Score"
                ]

            ]

        }

    )


    st.dataframe(

        comparison.style.format(

            {

                "Baseline Logistic":
                    "{:.2%}",

                "Logistic + SMOTE":
                    "{:.2%}"

            }

        ),

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # CHART
    # ========================================================

    st.markdown(
        "### Metric Comparison"
    )


    chart_data = comparison.set_index(
        "Metric"
    )


    st.bar_chart(
        chart_data
    )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.html(
        f"""
        <div class="card">

            <h3>
                What changed after SMOTE?
            </h3>

            <p>

                Accuracy changed from
                <strong>
                    {BASELINE_METRICS["Accuracy"]:.2%}
                </strong>
                to
                <strong>
                    {SMOTE_METRICS["Accuracy"]:.2%}
                </strong>.

            </p>

            <p>

                Recall improved from
                <strong>
                    {BASELINE_METRICS["Recall"]:.2%}
                </strong>
                to
                <strong>
                    {SMOTE_METRICS["Recall"]:.2%}
                </strong>.

            </p>

            <p>

                F1 Score improved from
                <strong>
                    {BASELINE_METRICS["F1 Score"]:.2%}
                </strong>
                to
                <strong>
                    {SMOTE_METRICS["F1 Score"]:.2%}
                </strong>.

            </p>

            <p>

                The SMOTE model therefore provided a stronger
                balance between precision and recall according
                to the F1 metric used for model selection.

            </p>

        </div>
        """
    )


# ============================================================
# PAGE 6 — METHODOLOGY
# ============================================================

elif page == "📘 Methodology":


    st.html(
        """
        <div class="section-title">
            📘 Methodology
        </div>

        <div class="section-description">
            A concise technical overview of how the customer-value
            prediction project was constructed.
        </div>
        """
    )


    # ========================================================
    # STEP 1
    # ========================================================

    st.html(
        """
        <div class="card">

            <h3>
                01 — Transaction Data
            </h3>

            <p>

                The project began with
                <strong>541,909 retail transactions</strong>.
                Transaction-level information was transformed
                into customer-level behavioural records.

            </p>

        </div>
        """
    )


    # ========================================================
    # STEP 2
    # ========================================================

    st.html(
        """
        <div class="card">

            <h3>
                02 — Temporal Feature Engineering
            </h3>

            <p>

                Historical transactions before
                <strong>2011-10-01</strong>
                were used to construct behavioural features.

                This creates a realistic modelling structure in
                which past behaviour is used to predict future
                customer value.

            </p>

        </div>
        """
    )


    # ========================================================
    # STEP 3
    # ========================================================

    st.html(
        """
        <div class="card">

            <h3>
                03 — Future Value Target
            </h3>

            <p>

                Future spending from
                <strong>2011-10-01</strong>
                onward was used to define the future customer
                value outcome.

                The reported future-spend median was
                <strong>£439.61</strong>.

            </p>

        </div>
        """
    )


    # ========================================================
    # STEP 4
    # ========================================================

    st.html(
        """
        <div class="card">

            <h3>
                04 — Model Development
            </h3>

            <p>

                Logistic Regression, Decision Tree,
                Random Forest and XGBoost were compared.

                SMOTE was applied to the training data to address
                class imbalance.

            </p>

        </div>
        """
    )


    # ========================================================
    # STEP 5
    # ========================================================

    st.html(
        """
        <div class="card">

            <h3>
                05 — Model Selection
            </h3>

            <p>

                Logistic Regression + SMOTE was selected based
                on the F1 Score among the evaluated modelling
                approaches.

            </p>

        </div>
        """
    )


    # ========================================================
    # STEP 6
    # ========================================================

    st.html(
        """
        <div class="card">

            <h3>
                06 — Deployment
            </h3>

            <p>

                The fitted preprocessing and Logistic Regression
                model were saved as a single
                <strong>customer_value_pipeline.pkl</strong>
                deployment pipeline.

            </p>

            <p>

                This allows the Streamlit application to receive
                raw behavioural inputs and pass them through the
                same preprocessing and prediction workflow.

            </p>

        </div>
        """
    )


    # ========================================================
    # LIMITATIONS
    # ========================================================

    st.html(
        """
        <div class="notice">

            <div class="notice-title">
                Important modelling limitations
            </div>

            Customer-value predictions are probabilistic estimates.
            Model performance depends on the underlying dataset,
            feature engineering, target definition and evaluation
            design.

            <br><br>

            A prediction should therefore be interpreted as a
            decision-support signal rather than a guaranteed
            statement about a customer's future behaviour.

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    f"""
    <div class="footer">

        <strong>
            ◈ Customer Value Intelligence
        </strong>

        <br><br>

        Behavioral Customer Value Prediction

        <br>

        RFM + behavioural analytics

        <br><br>

        Built with Python & Streamlit

        <br>

        Built by
        <strong>
            Olalemi Olaoluwakintan Emmanuel
        </strong>

    </div>
    """
)
