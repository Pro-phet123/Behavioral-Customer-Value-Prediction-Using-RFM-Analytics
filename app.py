import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Customer Value Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME / CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 85% 5%, rgba(99,102,241,.16), transparent 28%),
        radial-gradient(circle at 5% 25%, rgba(14,165,233,.10), transparent 25%),
        #080b14;
    color: #f8fafc;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.03em;
}

.hero {
    padding: 28px 30px;
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(255,255,255,.075), rgba(255,255,255,.025));
    box-shadow: 0 20px 60px rgba(0,0,0,.28);
    margin-bottom: 22px;
}

.eyebrow {
    color: #a5b4fc;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .16em;
    text-transform: uppercase;
}

.hero-title {
    font-size: clamp(2.2rem, 5vw, 4.3rem);
    line-height: .98;
    margin: 9px 0 14px;
}

.hero-copy {
    color: #aab4c8;
    font-size: 1.02rem;
    max-width: 850px;
    line-height: 1.65;
}

.metric-card {
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 18px;
    padding: 18px;
    background: rgba(255,255,255,.045);
    min-height: 118px;
}

.metric-label {
    color: #8995aa;
    font-size: .76rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    font-weight: 700;
}

.metric-value {
    font-family: 'Space Grotesk';
    font-size: 1.85rem;
    font-weight: 700;
    margin-top: 8px;
}

.metric-sub {
    color: #7f8ba1;
    font-size: .78rem;
    margin-top: 4px;
}

.result-high, .result-low {
    border-radius: 22px;
    padding: 28px;
    margin: 10px 0 20px;
    border: 1px solid rgba(255,255,255,.10);
}

.result-high {
    background: linear-gradient(135deg, rgba(34,197,94,.14), rgba(34,197,94,.035));
}

.result-low {
    background: linear-gradient(135deg, rgba(148,163,184,.12), rgba(148,163,184,.025));
}

.result-title {
    font-family: 'Space Grotesk';
    font-size: 2rem;
    font-weight: 700;
}

.prob {
    font-family: 'Space Grotesk';
    font-size: 3.4rem;
    font-weight: 700;
    line-height: 1;
}

.pill {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    background: rgba(165,180,252,.11);
    color: #c7d2fe;
    border: 1px solid rgba(165,180,252,.18);
    font-size: .76rem;
    font-weight: 700;
    margin-right: 5px;
}

.section-note {
    color: #8e9ab0;
    line-height: 1.6;
}

div[data-testid="stSidebar"] {
    background: rgba(5,8,16,.96);
    border-right: 1px solid rgba(255,255,255,.07);
}

div[data-testid="stButton"] > button {
    border-radius: 13px;
    min-height: 48px;
    font-weight: 700;
    border: 1px solid rgba(165,180,252,.25);
}

div[data-testid="stForm"] {
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 20px;
    padding: 22px;
    background: rgba(255,255,255,.035);
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


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

    The pipeline contains:
    1. StandardScaler
    2. Trained Logistic Regression model
    """
    if PIPELINE_PATH.exists():
        return joblib.load(PIPELINE_PATH)

    return None


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ◈ CVI")
    st.caption("Customer Value Intelligence")
    st.divider()

    page = st.radio(
        "Workspace",
        ["Prediction", "Customer Profile", "Model Context"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Model")
    st.markdown("""
    <span class="pill">Logistic Regression</span>
    <span class="pill">SMOTE</span>
    <span class="pill">Time-aware</span>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Behavioral Customer Value Prediction")
    st.caption("RFM + behavioural analytics")


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">Customer Intelligence • Predictive Analytics</div>
    <div class="hero-title">Know who is becoming<br>valuable <i>before</i> they do.</div>
    <div class="hero-copy">
        Turn historical purchasing behaviour into an actionable customer-value signal.
        Enter a customer's behavioural profile and estimate the likelihood that they
        belong to the high-value segment.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PREDICTION PAGE
# ============================================================
if page == "Prediction":

    if model is None:
        st.error(
            "Model pipeline not found. Add `customer_value_pipeline.pkl` "
            "to the application folder."
        )
        st.stop()

    st.markdown("### Customer prediction")

    st.markdown(
        '<p class="section-note">'
        'The prediction uses the same seven behavioural features used in the modelling notebook.'
        '</p>',
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):

        c1, c2, c3 = st.columns(3)

        # --------------------------------------------------------
        # COLUMN 1
        # --------------------------------------------------------
        with c1:
            recency = st.number_input(
                "Recency (days)",
                min_value=0.0,
                value=30.0,
                step=1.0
            )

            frequency = st.number_input(
                "Frequency (orders)",
                min_value=1.0,
                value=3.0,
                step=1.0
            )

            aov = st.number_input(
                "Average Order Value",
                min_value=0.0,
                value=150.0,
                step=10.0
            )

        # --------------------------------------------------------
        # COLUMN 2
        # --------------------------------------------------------
        with c2:
            avg_qty = st.number_input(
                "Average Quantity",
                min_value=0.0,
                value=10.0,
                step=1.0
            )

            total_items = st.number_input(
                "Total Items",
                min_value=0.0,
                value=30.0,
                step=1.0
            )

            unique_products = st.number_input(
                "Unique Products",
                min_value=0.0,
                value=15.0,
                step=1.0
            )

        # --------------------------------------------------------
        # COLUMN 3
        # --------------------------------------------------------
        with c3:
            transactions = st.number_input(
                "Number of Transactions",
                min_value=1.0,
                value=5.0,
                step=1.0
            )

            st.markdown("#### What this means")

            st.caption(
                "Lower recency generally indicates more recent activity, while "
                "higher frequency, product diversity and transaction volume indicate "
                "stronger historical engagement."
            )

        submitted = st.form_submit_button(
            "RUN CUSTOMER VALUE PREDICTION  →",
            use_container_width=True,
            type="primary",
        )

    # ============================================================
    # RUN PREDICTION
    # ============================================================
    if submitted:

        input_df = pd.DataFrame([{
            "Recency": recency,
            "Frequency": frequency,
            "Average_Order_Value": aov,
            "Average_Quantity": avg_qty,
            "Total_Items": total_items,
            "Unique_Products": unique_products,
            "Number_of_Transactions": transactions,
        }])[FEATURES]

        try:

            # The saved pipeline automatically performs:
            # Raw features → StandardScaler → Logistic Regression

            probability = float(
                model.predict_proba(input_df)[0, 1]
            )

            prediction = int(
                model.predict(input_df)[0]
            )

            st.divider()

            # ====================================================
            # HIGH-VALUE CUSTOMER
            # ====================================================
            if prediction == 1:

                st.markdown(f"""
                <div class="result-high">
                    <div class="eyebrow">Prediction • High Value</div>

                    <div class="result-title">
                        High-value customer signal detected.
                    </div>

                    <p class="section-note">
                        This customer shows behavioural characteristics associated with
                        the high-value segment in the trained model.
                    </p>

                    <div class="prob">
                        {probability:.1%}
                    </div>

                    <div class="metric-sub">
                        Estimated probability of high-value class
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ====================================================
            # LOWER-VALUE CUSTOMER
            # ====================================================
            else:

                st.markdown(f"""
                <div class="result-low">
                    <div class="eyebrow">Prediction • Lower Value</div>

                    <div class="result-title">
                        Lower high-value probability.
                    </div>

                    <p class="section-note">
                        The customer's current behavioural profile is less aligned with
                        the high-value segment.
                    </p>

                    <div class="prob">
                        {probability:.1%}
                    </div>

                    <div class="metric-sub">
                        Estimated probability of high-value class
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ====================================================
            # PROBABILITY GAUGE
            # ====================================================
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%"},
                    title={"text": "High-value probability"},
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
                    b=20
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    color="#e2e8f0"
                ),
            )

            left, right = st.columns([1.15, 1])

            # ----------------------------------------------------
            # GAUGE
            # ----------------------------------------------------
            with left:
                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # ----------------------------------------------------
            # BEHAVIOURAL SNAPSHOT
            # ----------------------------------------------------
            with right:

                st.markdown("### Behavioural snapshot")

                profile = pd.DataFrame({
                    "Feature": FEATURES,
                    "Customer": input_df.iloc[0].values
                })

                st.dataframe(
                    profile,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Prediction is a model-based probability, not a guarantee of future spending."
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# CUSTOMER PROFILE PAGE
# ============================================================
elif page == "Customer Profile":

    st.markdown("### Customer profile simulator")

    st.markdown(
        '<p class="section-note">'
        'Use this space to reason about how behavioural changes may affect customer value.'
        '</p>',
        unsafe_allow_html=True
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

            values[feature] = st.slider(
                feature.replace("_", " "),
                min_value=0.0,
                max_value=float(
                    max(
                        base[feature] * 5,
                        100
                    )
                ),
                value=float(
                    base[feature]
                ),
            )

    profile = pd.DataFrame([values])

    st.markdown("### Current profile")

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "This simulator is intended for behavioural exploration. "
        "The production prediction page uses the saved model pipeline."
    )


# ============================================================
# MODEL CONTEXT PAGE
# ============================================================
elif page == "Model Context":

    st.markdown("### What is behind the prediction?")

    m1, m2, m3, m4 = st.columns(4)

    metrics = [
        ("541,909", "Raw transactions"),
        ("3,616", "Customer-level records"),
        ("7", "Behavioural features"),
        ("£439.61", "Future-spend median"),
    ]

    for col, (value, label) in zip(
        [m1, m2, m3, m4],
        metrics
    ):

        with col:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### The modelling logic")

    st.markdown("""
    <div class="hero">

        <div class="eyebrow">
            Leakage-aware design
        </div>

        <h3>
            Past behaviour → prediction → future outcome
        </h3>

        <p class="section-note">
            Historical transactions before <b>2011-10-01</b> were used to engineer
            customer behaviour. Future transactions from <b>2011-10-01</b> onward
            were used to define future spending and the high-value target.
        </p>

        <p class="section-note">
            The modelling workflow compared Logistic Regression, Decision Tree,
            Random Forest and XGBoost, with SMOTE applied to the training data.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Key evaluation result")

    result_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Baseline Logistic": [
            0.7873,
            0.7297,
            0.3951,
            0.5127
        ],

        "Logistic + SMOTE": [
            0.7652,
            0.5792,
            0.6244,
            0.6009
        ],
    })

    fig = px.bar(
        result_df,
        x="Metric",
        y=[
            "Baseline Logistic",
            "Logistic + SMOTE"
        ],
        barmode="group",
        range_y=[0, 1],
        labels={
            "value": "Score",
            "variable": "Model"
        }
    )

    fig.update_layout(
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.025)",
        font=dict(
            color="#e2e8f0"
        ),
        legend_title_text="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### Important modelling note")

    st.warning(
        "The deployed application uses a single sklearn Pipeline containing "
        "the fitted StandardScaler and the trained Logistic Regression model. "
        "This prevents preprocessing mismatch between the notebook and the web app."
    )


# ============================================================
# FOOTER
# ============================================================
st.divider()

st.caption(
    "Behavioral Customer Value Prediction • RFM + behavioural analytics • "
    "Built with Python & Streamlit"
)