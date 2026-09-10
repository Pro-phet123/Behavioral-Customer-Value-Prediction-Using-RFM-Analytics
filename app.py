import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


# ============================================================
# CREATE CUSTOM 3D RFM CUBE FAVICON
# ============================================================

FAVICON_PATH = Path("rfm_cube.svg")

if not FAVICON_PATH.exists():
    favicon_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">
        <defs>
            <linearGradient id="top" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#a5b4fc"/>
                <stop offset="100%" stop-color="#6366f1"/>
            </linearGradient>

            <linearGradient id="left" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#6366f1"/>
                <stop offset="100%" stop-color="#312e81"/>
            </linearGradient>

            <linearGradient id="right" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#38bdf8"/>
                <stop offset="100%" stop-color="#2563eb"/>
            </linearGradient>

            <filter id="shadow">
                <feDropShadow
                    dx="0"
                    dy="7"
                    stdDeviation="6"
                    flood-color="#000000"
                    flood-opacity=".35"
                />
            </filter>
        </defs>

        <rect
            width="128"
            height="128"
            rx="28"
            fill="#080b14"
        />

        <g filter="url(#shadow)">
            <!-- TOP -->
            <polygon
                points="64,17 108,39 64,61 20,39"
                fill="url(#top)"
            />

            <!-- LEFT -->
            <polygon
                points="20,39 64,61 64,111 20,87"
                fill="url(#left)"
            />

            <!-- RIGHT -->
            <polygon
                points="108,39 64,61 64,111 108,87"
                fill="url(#right)"
            />

            <!-- TOP R -->
            <text
                x="52"
                y="39"
                font-family="Arial, sans-serif"
                font-size="18"
                font-weight="700"
                fill="#ffffff"
                text-anchor="middle"
            >R</text>

            <!-- LEFT F -->
            <text
                x="40"
                y="75"
                font-family="Arial, sans-serif"
                font-size="18"
                font-weight="700"
                fill="#ffffff"
                text-anchor="middle"
            >F</text>

            <!-- RIGHT M -->
            <text
                x="84"
                y="75"
                font-family="Arial, sans-serif"
                font-size="18"
                font-weight="700"
                fill="#ffffff"
                text-anchor="middle"
            >M</text>

            <!-- Cube highlights -->
            <line
                x1="64"
                y1="61"
                x2="64"
                y2="108"
                stroke="rgba(255,255,255,.22)"
                stroke-width="2"
            />

            <line
                x1="22"
                y1="40"
                x2="63"
                y2="61"
                stroke="rgba(255,255,255,.18)"
                stroke-width="2"
            />

            <line
                x1="106"
                y1="40"
                x2="65"
                y2="61"
                stroke="rgba(255,255,255,.18)"
                stroke-width="2"
            />
        </g>
    </svg>
    """

    try:
        FAVICON_PATH.write_text(favicon_svg, encoding="utf-8")
    except Exception:
        pass


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Value Intelligence",
    page_icon=str(FAVICON_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PREMIUM DARK THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html,
    body,
    [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(99, 102, 241, 0.16),
                transparent 28%
            ),
            radial-gradient(
                circle at 5% 25%,
                rgba(14, 165, 233, 0.10),
                transparent 25%
            ),
            #080b14;

        color: #f8fafc;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1,
    h2,
    h3,
    h4 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.03em;
    }

    /* ======================================================
       SIDEBAR
       ====================================================== */

    div[data-testid="stSidebar"] {
        background: rgba(5, 8, 16, 0.97);
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }

    div[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ======================================================
       HERO
       ====================================================== */

    .hero {
        padding: 32px 34px;
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 26px;

        background:
            linear-gradient(
                135deg,
                rgba(255, 255, 255, 0.075),
                rgba(255, 255, 255, 0.025)
            );

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.28);

        margin-bottom: 24px;
    }

    .eyebrow {
        color: #a5b4fc;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.2rem, 5vw, 4.3rem);
        line-height: 0.98;
        letter-spacing: -0.055em;
        margin: 10px 0 16px;
        color: #f8fafc;
    }

    .hero-title i {
        color: #a5b4fc;
    }

    .hero-copy {
        color: #aab4c8;
        font-size: 1.02rem;
        max-width: 850px;
        line-height: 1.7;
    }

    /* ======================================================
       METRIC CARDS
       ====================================================== */

    .metric-card {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.045);
        min-height: 120px;
        transition: all 0.25s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(165, 180, 252, 0.25);
        background: rgba(255, 255, 255, 0.065);
    }

    .metric-label {
        color: #8995aa;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        margin-top: 8px;
        color: #f8fafc;
    }

    .metric-sub {
        color: #7f8ba1;
        font-size: 0.78rem;
        margin-top: 5px;
    }

    /* ======================================================
       SECTION TEXT
       ====================================================== */

    .section-note {
        color: #8e9ab0;
        line-height: 1.65;
    }

    /* ======================================================
       PILLS
       ====================================================== */

    .pill {
        display: inline-block;
        padding: 6px 11px;
        margin: 3px 4px 3px 0;

        border-radius: 999px;

        background: rgba(165, 180, 252, 0.11);
        color: #c7d2fe;

        border: 1px solid rgba(165, 180, 252, 0.18);

        font-size: 0.74rem;
        font-weight: 700;
    }

    /* ======================================================
       RESULT CARDS
       ====================================================== */

    .result-high,
    .result-low {
        border-radius: 22px;
        padding: 28px;

        margin: 10px 0 20px;

        border: 1px solid rgba(255, 255, 255, 0.10);

        box-shadow:
            0 16px 45px rgba(0, 0, 0, 0.18);
    }

    .result-high {
        background:
            linear-gradient(
                135deg,
                rgba(34, 197, 94, 0.14),
                rgba(34, 197, 94, 0.035)
            );
    }

    .result-low {
        background:
            linear-gradient(
                135deg,
                rgba(148, 163, 184, 0.12),
                rgba(148, 163, 184, 0.025)
            );
    }

    .result-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 6px;
    }

    .prob {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.4rem;
        font-weight: 700;
        line-height: 1;
        margin-top: 20px;
    }

    /* ======================================================
       FORM
       ====================================================== */

    div[data-testid="stForm"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;

        background: rgba(255, 255, 255, 0.035);
    }

    /* ======================================================
       INPUTS
       ====================================================== */

    div[data-baseweb="input"] {
        background: rgba(255, 255, 255, 0.035);
        border-radius: 10px;
    }

    div[data-baseweb="select"] {
        border-radius: 10px;
    }

    /* ======================================================
       BUTTONS
       ====================================================== */

    div[data-testid="stButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 13px;
        min-height: 48px;
        font-weight: 700;

        border: 1px solid rgba(165, 180, 252, 0.25);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    div[data-testid="stButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 8px 25px rgba(99, 102, 241, 0.18);
    }

    /* ======================================================
       DATAFRAME
       ====================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    /* ======================================================
       PLOTLY
       ====================================================== */

    .plot-container {
        border-radius: 18px;
    }

    /* ======================================================
       ALERTS
       ====================================================== */

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* ======================================================
       MOBILE RESPONSIVENESS
       ====================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        .hero {
            padding: 24px 20px;
            border-radius: 20px;
        }

        .hero-title {
            font-size: 2.45rem;
        }

        .hero-copy {
            font-size: 0.94rem;
        }

        .metric-card {
            min-height: 105px;
            padding: 16px;
        }

        .metric-value {
            font-size: 1.5rem;
        }

        .result-high,
        .result-low {
            padding: 22px;
        }

        .result-title {
            font-size: 1.55rem;
        }

        .prob {
            font-size: 2.7rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

PIPELINE_PATH = Path("customer_value_pipeline.pkl")

FEATURES = [
    "Recency",
    "Frequency",
    "Average_Order_Value",
    "Average_Quantity",
    "Total_Items",
    "Unique_Products",
    "Number_of_Transactions",
]


@st.cache_resource
def load_model():
    """
    Load the saved deployment pipeline.

    Expected pipeline:

    Raw Features
        ↓
    StandardScaler
        ↓
    Logistic Regression
    """

    if PIPELINE_PATH.exists():
        return joblib.load(PIPELINE_PATH)

    return None


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.65rem;
            font-weight: 700;
            letter-spacing: -0.04em;
        ">
            ◈ CVI
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Customer Value Intelligence")

    st.divider()

    page = st.radio(
        "Workspace",
        [
            "Prediction",
            "Customer Profile",
            "Model Context",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### Model")

    st.markdown(
        """
        <span class="pill">Logistic Regression</span>
        <span class="pill">SMOTE</span>
        <span class="pill">Time-aware</span>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("Behavioral Customer Value Prediction")
    st.caption("RFM + behavioural analytics")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="eyebrow">
            Customer Intelligence • Predictive Analytics
        </div>

        <div class="hero-title">
            Know who is becoming<br>
            valuable <i>before</i> they do.
        </div>

        <div class="hero-copy">
            Turn historical purchasing behaviour into an actionable
            customer-value signal. Enter a customer's behavioural profile
            and estimate the likelihood that they belong to the high-value
            segment.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PREDICTION PAGE
# ============================================================

if page == "Prediction":

    if model is None:

        st.error(
            "Model pipeline not found. Add "
            "`customer_value_pipeline.pkl` "
            "to the application folder."
        )

        st.stop()

    st.markdown("### Customer prediction")

    st.markdown(
        """
        <p class="section-note">
            The prediction uses the same seven behavioural features
            used in the modelling notebook.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # PREDICTION FORM
    # ========================================================

    with st.form("prediction_form"):

        c1, c2, c3 = st.columns(3)

        # ----------------------------------------------------
        # COLUMN 1
        # ----------------------------------------------------

        with c1:

            recency = st.number_input(
                "Recency (days)",
                min_value=0.0,
                value=30.0,
                step=1.0,
                help="Number of days since the customer's last purchase.",
            )

            frequency = st.number_input(
                "Frequency (orders)",
                min_value=1.0,
                value=3.0,
                step=1.0,
                help="Number of customer orders.",
            )

            aov = st.number_input(
                "Average Order Value",
                min_value=0.0,
                value=150.0,
                step=10.0,
                help="Average monetary value of each order.",
            )

        # ----------------------------------------------------
        # COLUMN 2
        # ----------------------------------------------------

        with c2:

            avg_qty = st.number_input(
                "Average Quantity",
                min_value=0.0,
                value=10.0,
                step=1.0,
                help="Average number of items purchased per transaction.",
            )

            total_items = st.number_input(
                "Total Items",
                min_value=0.0,
                value=30.0,
                step=1.0,
                help="Total number of items purchased.",
            )

            unique_products = st.number_input(
                "Unique Products",
                min_value=0.0,
                value=15.0,
                step=1.0,
                help="Number of distinct products purchased.",
            )

        # ----------------------------------------------------
        # COLUMN 3
        # ----------------------------------------------------

        with c3:

            transactions = st.number_input(
                "Number of Transactions",
                min_value=1.0,
                value=5.0,
                step=1.0,
                help="Total number of transactions.",
            )

            st.markdown("#### What this means")

            st.caption(
                "Lower recency generally indicates more recent activity, "
                "while higher frequency, product diversity and transaction "
                "volume indicate stronger historical engagement."
            )

        submitted = st.form_submit_button(
            "RUN CUSTOMER VALUE PREDICTION  →",
            use_container_width=True,
            type="primary",
        )


    # ========================================================
    # RUN PREDICTION
    # ========================================================

    if submitted:

        input_df = pd.DataFrame(
            [
                {
                    "Recency": recency,
                    "Frequency": frequency,
                    "Average_Order_Value": aov,
                    "Average_Quantity": avg_qty,
                    "Total_Items": total_items,
                    "Unique_Products": unique_products,
                    "Number_of_Transactions": transactions,
                }
            ]
        )[FEATURES]

        try:

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            if hasattr(model, "predict_proba"):

                probability = float(
                    model.predict_proba(input_df)[0, 1]
                )

            else:

                prediction_raw = int(
                    model.predict(input_df)[0]
                )

                probability = float(prediction_raw)


            prediction = int(
                model.predict(input_df)[0]
            )

            st.divider()

            # =================================================
            # HIGH VALUE
            # =================================================

            if prediction == 1:

                st.markdown(
                    f"""
                    <div class="result-high">

                        <div class="eyebrow">
                            Prediction • High Value
                        </div>

                        <div class="result-title">
                            High-value customer signal detected.
                        </div>

                        <p class="section-note">
                            This customer shows behavioural characteristics
                            associated with the high-value segment in the
                            trained model.
                        </p>

                        <div class="prob">
                            {probability:.1%}
                        </div>

                        <div class="metric-sub">
                            Estimated probability of high-value class
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # =================================================
            # LOWER VALUE
            # =================================================

            else:

                st.markdown(
                    f"""
                    <div class="result-low">

                        <div class="eyebrow">
                            Prediction • Lower Value
                        </div>

                        <div class="result-title">
                            Lower high-value probability.
                        </div>

                        <p class="section-note">
                            The customer's current behavioural profile is
                            less aligned with the high-value segment.
                        </p>

                        <div class="prob">
                            {probability:.1%}
                        </div>

                        <div class="metric-sub">
                            Estimated probability of high-value class
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            # =================================================
            # PROBABILITY GAUGE
            # =================================================

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,

                    number={
                        "suffix": "%",
                        "font": {
                            "size": 42
                        },
                    },

                    title={
                        "text": "High-value probability",
                        "font": {
                            "size": 18
                        },
                    },

                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },

                        "bar": {
                            "thickness": 0.25
                        },

                        "steps": [
                            {
                                "range": [0, 50]
                            },
                            {
                                "range": [50, 100]
                            },
                        ],

                        "threshold": {
                            "line": {
                                "width": 4
                            },

                            "thickness": 0.8,

                            "value": 50,
                        },
                    },
                )
            )

            fig.update_layout(
                height=300,

                margin=dict(
                    l=25,
                    r=25,
                    t=55,
                    b=20,
                ),

                paper_bgcolor="rgba(0,0,0,0)",

                font=dict(
                    color="#e2e8f0",
                ),
            )


            # =================================================
            # GAUGE + PROFILE
            # =================================================

            left, right = st.columns(
                [1.15, 1]
            )

            # -------------------------------------------------
            # GAUGE
            # -------------------------------------------------

            with left:

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            # -------------------------------------------------
            # BEHAVIOURAL SNAPSHOT
            # -------------------------------------------------

            with right:

                st.markdown(
                    "### Behavioural snapshot"
                )

                profile = pd.DataFrame(
                    {
                        "Feature": FEATURES,
                        "Customer": input_df.iloc[0].values,
                    }
                )

                st.dataframe(
                    profile,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Prediction is a model-based probability, "
                    "not a guarantee of future spending."
                )


        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# CUSTOMER PROFILE PAGE
# ============================================================

elif page == "Customer Profile":

    st.markdown(
        "### Customer profile simulator"
    )

    st.markdown(
        """
        <p class="section-note">
            Use this space to reason about how behavioural changes
            may affect customer value.
        </p>
        """,
        unsafe_allow_html=True,
    )

    base = {
        "Recency": 30,
        "Frequency": 3,
        "Average_Order_Value": 150,
        "Average_Quantity": 10,
        "Total_Items": 30,
        "Unique_Products": 15,
        "Number_of_Transactions": 5,
    }

    cols = st.columns(2)

    values = {}

    for i, feature in enumerate(FEATURES):

        with cols[i % 2]:

            maximum = float(
                max(
                    base[feature] * 5,
                    100,
                )
            )

            values[feature] = st.slider(
                feature.replace("_", " "),
                min_value=0.0,
                max_value=maximum,
                value=float(base[feature]),
            )

    profile = pd.DataFrame(
        [values]
    )

    st.markdown(
        "### Current profile"
    )

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "This simulator is intended for behavioural exploration. "
        "The production prediction page uses the saved model pipeline."
    )


# ============================================================
# MODEL CONTEXT PAGE
# ============================================================

elif page == "Model Context":

    st.markdown(
        "### What is behind the prediction?"
    )

    # ========================================================
    # TOP METRICS
    # ========================================================

    m1, m2, m3, m4 = st.columns(4)

    metrics = [
        ("541,909", "Raw transactions"),
        ("3,616", "Customer-level records"),
        ("7", "Behavioural features"),
        ("£439.61", "Future-spend median"),
    ]

    for col, (value, label) in zip(
        [m1, m2, m3, m4],
        metrics,
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="metric-value">
                        {value}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


    # ========================================================
    # MODELLING LOGIC
    # ========================================================

    st.markdown(
        "### The modelling logic"
    )

    st.markdown(
        """
        <div class="hero">

            <div class="eyebrow">
                Leakage-aware design
            </div>

            <h3>
                Past behaviour → prediction → future outcome
            </h3>

            <p class="section-note">
                Historical transactions before
                <b>2011-10-01</b>
                were used to engineer customer behaviour.
                Future transactions from
                <b>2011-10-01</b>
                onward were used to define future spending
                and the high-value target.
            </p>

            <p class="section-note">
                The modelling workflow compared Logistic Regression,
                Decision Tree, Random Forest and XGBoost, with SMOTE
                applied to the training data.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # EVALUATION
    # ========================================================

    st.markdown(
        "### Key evaluation results"
    )

    result_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
            ],

            "Baseline Logistic": [
                0.7873,
                0.7297,
                0.3951,
                0.5127,
            ],

            "Logistic + SMOTE": [
                0.7652,
                0.5792,
                0.6244,
                0.6009,
            ],
        }
    )


    # ========================================================
    # BAR CHART
    # ========================================================

    fig = px.bar(
        result_df,
        x="Metric",
        y=[
            "Baseline Logistic",
            "Logistic + SMOTE",
        ],

        barmode="group",

        range_y=[0, 1],

        labels={
            "value": "Score",
            "variable": "Model",
        },
    )

    fig.update_layout(
        height=390,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(255,255,255,.025)",

        font=dict(
            color="#e2e8f0"
        ),

        legend_title_text="",

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20,
        ),
    )

    fig.update_yaxes(
        tickformat=".0%",
        gridcolor="rgba(255,255,255,.06)",
    )

    fig.update_xaxes(
        gridcolor="rgba(255,255,255,.04)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ========================================================
    # MODELLING NOTE
    # ========================================================

    st.markdown(
        "### Important modelling note"
    )

    st.warning(
        "The deployed application uses a single sklearn Pipeline "
        "containing the fitted StandardScaler and the trained "
        "Logistic Regression model. This prevents preprocessing "
        "mismatch between the notebook and the web app."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        color: #667085;
        font-size: 0.78rem;
        padding: 10px 0 20px;
    ">
        Behavioral Customer Value Prediction
        • RFM + behavioural analytics
        • Built with Python & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
