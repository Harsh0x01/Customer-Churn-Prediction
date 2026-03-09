import streamlit as st
import joblib
import numpy as np
import shap
import plotly.graph_objects as go

st.set_page_config(
    page_title="ChurnIQ",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: -apple-system, 'Segoe UI', system-ui, sans-serif;
    background: #0a0e17;
    color: #94a3b8;
    font-size: 14px;
}

.stApp { background: #0a0e17; }

header, footer, #MainMenu { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
div[data-testid="stVerticalBlock"] { gap: 0 !important; }
.element-container { margin-bottom: 8px !important; }

/* NAVBAR */
.navbar {
    height: 50px;
    background: #0d1220;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
}
.nav-brand {
    font-size: 0.88rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.5px;
}
.nav-brand span { color: #4d9eff; }
.nav-center {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.15);
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.nav-right {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.65rem;
    color: rgba(255,255,255,0.15);
    letter-spacing: 1px;
}
.live-dot {
    width: 6px; height: 6px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 5px #22c55e;
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }

/* PANEL TITLE */
.panel-title {
    font-size: 0.6rem;
    font-weight: 600;
    color: rgba(255,255,255,0.18);
    letter-spacing: 2px;
    text-transform: uppercase;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 14px;
}

/* PROB CARD */
.prob-card {
    background: #0d1220;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.prob-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.prob-card.high::before  { background: #ef4444; }
.prob-card.medium::before { background: #f97316; }
.prob-card.low::before   { background: #22c55e; }

.prob-eyebrow {
    font-size: 0.6rem;
    color: rgba(255,255,255,0.2);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.prob-number {
    font-size: 3.6rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -1px;
    font-variant-numeric: tabular-nums;
}
.prob-number.high   { color: #ef4444; }
.prob-number.medium { color: #f97316; }
.prob-number.low    { color: #22c55e; }

.prob-badge {
    display: inline-block;
    margin-top: 10px;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.prob-badge.high   { background: rgba(239,68,68,0.1);  color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }
.prob-badge.medium { background: rgba(249,115,22,0.1); color: #f97316; border: 1px solid rgba(249,115,22,0.2); }
.prob-badge.low    { background: rgba(34,197,94,0.1);  color: #22c55e; border: 1px solid rgba(34,197,94,0.2); }

/* STATS CARD */
.stats-card {
    background: #0d1220;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 16px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    height: 100%;
}
.stat-value {
    font-size: 1.2rem;
    font-weight: 600;
    color: #e2e8f0;
    line-height: 1.2;
    font-variant-numeric: tabular-nums;
}
.stat-label {
    font-size: 0.58rem;
    color: rgba(255,255,255,0.18);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* SECTION HEADER */
.section-hdr {
    font-size: 0.6rem;
    font-weight: 600;
    color: rgba(255,255,255,0.18);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 14px 0 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-hdr::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.05);
}

/* RETENTION */
.ret-item {
    background: #0d1220;
    border: 1px solid rgba(255,255,255,0.05);
    border-left: 2px solid #4d9eff;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin-bottom: 6px;
}
.ret-tag {
    font-size: 0.58rem;
    color: #4d9eff;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 3px;
    font-weight: 600;
}
.ret-body {
    font-size: 0.78rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* STABLE */
.stable-card {
    background: #0d1220;
    border: 1px solid rgba(34,197,94,0.12);
    border-radius: 8px;
    padding: 24px;
    text-align: center;
    margin-top: 6px;
}
.stable-check {
    font-size: 1.4rem;
    color: #22c55e;
    margin-bottom: 6px;
}
.stable-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #22c55e;
    letter-spacing: 0.5px;
}
.stable-sub {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.18);
    margin-top: 5px;
    line-height: 1.6;
}

/* IDLE */
.idle-card {
    background: #0d1220;
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 50px 24px;
    text-align: center;
}
.idle-title {
    font-size: 0.68rem;
    color: rgba(255,255,255,0.1);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 10px;
}

/* MODEL FOOTER */
.model-footer {
    padding: 12px 0 0;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-size: 0.6rem;
    color: rgba(255,255,255,0.1);
    line-height: 2;
    letter-spacing: 0.3px;
}

/* SHAP LEGEND */
.shap-legend {
    font-size: 0.62rem;
    color: rgba(255,255,255,0.18);
    letter-spacing: 0.5px;
    margin-bottom: 4px;
    display: flex;
    gap: 14px;
}
.leg-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 4px;
    vertical-align: middle;
}

/* STREAMLIT OVERRIDES */
div[data-testid="stNumberInput"] input {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 5px !important;
    color: #e2e8f0 !important;
    font-size: 0.82rem !important;
    padding: 7px 10px !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #4d9eff !important;
    box-shadow: 0 0 0 2px rgba(77,158,255,0.12) !important;
    outline: none !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 5px !important;
    color: #e2e8f0 !important;
    font-size: 0.82rem !important;
}
.stSlider > div > div > div > div { background: #4d9eff !important; }
.stSlider > div > div > div { background: rgba(255,255,255,0.07) !important; }
label {
    color: rgba(255,255,255,0.25) !important;
    font-size: 0.65rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
    text-transform: uppercase !important;
    margin-bottom: 2px !important;
}
.stButton > button {
    width: 100% !important;
    background: #4d9eff !important;
    border: none !important;
    border-radius: 6px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.3px !important;
    padding: 11px 24px !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #3b8aef !important; }
</style>
""", unsafe_allow_html=True)

# LOAD MODELS
@st.cache_resource
def load_models():
    model         = joblib.load('model/xgb_model.pkl')
    scaler        = joblib.load('model/scaler.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_models()

# RETENTION ENGINE
def retention_engine(shap_vals, feature_names):
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    action_map = {
        'Age':            ('Age Profile Risk',       'Assign a dedicated senior relationship manager with a personalized financial plan'),
        'NumOfProducts':  ('Product Usage Anomaly',  'Offer a tailored product bundle with exclusive loyalty pricing and an upgrade path'),
        'IsActiveMember': ('Inactivity Detected',    'Launch a re-engagement campaign with cashback rewards and a personal check-in call'),
        'Balance':        ('High Balance at Risk',   'Upgrade to premium tier with better interest rates, wealth management, and priority support'),
        'Tenure':         ('Low Tenure Risk',        'Activate early loyalty program with milestone rewards and long-term relationship incentives'),
        'Geography':      ('Regional Churn Signal',  'Route to regional retention specialist for a localized and personalized intervention'),
        'Gender':         ('Demographic Signal',     'Assign a financial advisor with targeted product offers matched to customer profile'),
        'CreditScore':    ('Credit Profile Risk',    'Enroll in credit improvement program with score-linked account benefits and rewards'),
        'EstimatedSalary':('Salary Bracket Signal',  'Offer a salary-linked savings account with premium rates and automatic investing'),
        'HasCrCard':      ('Card Usage Flag',        'Upgrade credit card with enhanced cashback, travel rewards, and zero annual fee'),
    }
    results = []
    for idx in top_idx:
        fname = feature_names[idx]
        if fname in action_map:
            results.append(action_map[fname])
    return results

# SHAP CHART
def shap_chart(shap_vals, feature_names):
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:6]
    feats   = [feature_names[i] for i in top_idx][::-1]
    vals    = [shap_vals[i]     for i in top_idx][::-1]
    colors  = ['#ef4444' if v > 0 else '#4d9eff' for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate='%{x:.3f}<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,18,32,0.5)',
        height=200,
        margin=dict(t=4, b=4, l=4, r=12),
        font=dict(color='#475569', size=10),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.04)',
            zerolinecolor='rgba(255,255,255,0.08)',
            tickfont=dict(color='#475569', size=9),
            tickformat='.2f'
        ),
        yaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(color='#94a3b8', size=10)),
        bargap=0.32
    )
    return fig

# NAVBAR
st.markdown("""
<div class="navbar">
    <div class="nav-brand">Churn<span>IQ</span></div>
    <div class="nav-center">XGBoost · SHAP Explainability · Retention Engine</div>
    <div class="nav-right">
        <div class="live-dot"></div>
        Model Online &nbsp;·&nbsp; AUC 0.85 &nbsp;·&nbsp; CV 0.96
    </div>
</div>
""", unsafe_allow_html=True)

# LAYOUT
left, right = st.columns([0.78, 1.22], gap="small")

# LEFT
with left:
    st.markdown('<div style="padding:18px 18px 0">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Customer Profile</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        age          = st.number_input("Age", 18, 100, 38)
        balance      = st.number_input("Balance ($)", 0.0, 300000.0, 75000.0, 1000.0)
        has_cr_card  = st.selectbox("Credit Card", ["Yes", "No"])
        num_products = st.selectbox("Products", [1, 2, 3, 4])
    with c2:
        geography    = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender       = st.selectbox("Gender", ["Male", "Female"])
        tenure       = st.slider("Tenure (yrs)", 0, 10, 5)
        is_active    = st.selectbox("Active Member", ["Yes", "No"])
        salary       = st.number_input("Salary ($)", 0.0, 250000.0, 100000.0, 1000.0)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    predict_btn = st.button("Run Analysis", use_container_width=True)

    st.markdown("""
    <div class="model-footer">
        Model &nbsp;·&nbsp; XGBoost Classifier &nbsp;&nbsp;
        Accuracy &nbsp;·&nbsp; 85.15% &nbsp;&nbsp;
        ROC-AUC &nbsp;·&nbsp; 0.85 &nbsp;&nbsp;
        CV &nbsp;·&nbsp; 0.96 ± 0.04 &nbsp;&nbsp;
        Threshold &nbsp;·&nbsp; 0.35
    </div>
    </div>
    """, unsafe_allow_html=True)

# RIGHT
with right:
    st.markdown('<div style="padding:18px 24px 0">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Analysis Output</div>', unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div class="idle-card">
            <div style="font-size:1.6rem; opacity:0.1; color:#fff">—</div>
            <div class="idle-title">Awaiting customer data</div>
            <div style="font-size:0.65rem; color:rgba(255,255,255,0.07); margin-top:5px; letter-spacing:1px;">
                Configure inputs and run analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Running analysis..."):
            geo_map    = {"France": 0, "Germany": 1, "Spain": 2}
            gender_map = {"Male": 1, "Female": 0}
            input_data = np.array([[
                credit_score, geo_map[geography], gender_map[gender],
                age, tenure, balance, num_products,
                1 if has_cr_card == "Yes" else 0,
                1 if is_active   == "Yes" else 0,
                salary
            ]])
            input_scaled = scaler.transform(input_data)
            prob         = model.predict_proba(input_scaled)[0][1]
            prediction   = int(prob >= 0.35)
            explainer    = shap.TreeExplainer(model)
            shap_values  = explainer.shap_values(input_scaled)
            shap_vals    = shap_values[0]

        if prob >= 0.65:
            risk_cls = "high"
            risk_txt = "High Risk — Immediate Action Required"
        elif prob >= 0.35:
            risk_cls = "medium"
            risk_txt = "Medium Risk — Monitor Closely"
        else:
            risk_cls = "low"
            risk_txt = "Low Risk — Customer Stable"

        h1, h2 = st.columns(2)
        with h1:
            st.markdown(f"""
            <div class="prob-card {risk_cls}">
                <div class="prob-eyebrow">Churn Probability</div>
                <div class="prob-number {risk_cls}">{prob*100:.1f}<span style="font-size:1.6rem;font-weight:300">%</span></div>
                <div class="prob-badge {risk_cls}">{risk_txt}</div>
            </div>
            """, unsafe_allow_html=True)

        with h2:
            ac = "#22c55e" if is_active == "Yes" else "#ef4444"
            av = "Active" if is_active == "Yes" else "Inactive"
            st.markdown(f"""
            <div class="stats-card">
                <div><div class="stat-value">{age}</div><div class="stat-label">Age</div></div>
                <div><div class="stat-value">{tenure}y</div><div class="stat-label">Tenure</div></div>
                <div><div class="stat-value">{credit_score}</div><div class="stat-label">Credit</div></div>
                <div><div class="stat-value">{num_products}</div><div class="stat-label">Products</div></div>
                <div><div class="stat-value" style="font-size:0.85rem">{geography[:3].upper()}</div><div class="stat-label">Region</div></div>
                <div><div class="stat-value" style="font-size:0.82rem;color:{ac}">{av}</div><div class="stat-label">Status</div></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-hdr">SHAP Feature Impact</div>', unsafe_allow_html=True)
        st.markdown('<div class="shap-legend"><span><span class="leg-dot" style="background:#ef4444"></span>Increases churn</span><span><span class="leg-dot" style="background:#4d9eff"></span>Reduces churn</span></div>', unsafe_allow_html=True)
        st.plotly_chart(shap_chart(shap_vals, feature_names), use_container_width=True, config={'displayModeBar': False})

        if prediction == 1:
            actions = retention_engine(shap_vals, feature_names)
            st.markdown('<div class="section-hdr">Retention Strategy</div>', unsafe_allow_html=True)
            for i, (tag, action) in enumerate(actions):
                st.markdown(f"""
                <div class="ret-item">
                    <div class="ret-tag">Action {i+1:02d} · {tag}</div>
                    <div class="ret-body">{action}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stable-card">
                <div class="stable-check">&#10003;</div>
                <div class="stable-title">Customer Stable</div>
                <div class="stable-sub">Churn probability is below the intervention threshold.<br>Continue standard engagement — no immediate action required.</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)