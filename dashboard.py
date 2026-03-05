# =============================================================================
# MAIZE YIELD PREDICTION DASHBOARD
# Uasin Gishu County, Kenya — KCA Tech Expo | March 2026
# =============================================================================
# Run with:
#   source /home/caleb/venv/bin/activate
#   streamlit run /home/caleb/Desktop/Maize_Yield_Submission/dashboard.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Maize Yield Prediction — Uasin Gishu",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# STYLING
# =============================================================================

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f5f5f0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a3c2e;
    }
    section[data-testid="stSidebar"] * {
        color: #e8f5e9 !important;
    }
    section[data-testid="stSidebar"] .stSlider label {
        color: #a5d6a7 !important;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* Headers */
    h1 { color: #1a3c2e !important; font-family: Georgia, serif !important; }
    h2 { color: #1a3c2e !important; }
    h3 { color: #2d6a4f !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: white;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #555;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d6a4f !important;
        color: white !important;
    }

    /* Prediction result box */
    .prediction-box {
        background: linear-gradient(135deg, #1a3c2e, #2d6a4f);
        border-radius: 12px;
        padding: 28px;
        text-align: center;
        color: white;
        margin: 16px 0;
    }
    .prediction-box h1 {
        color: white !important;
        font-size: 3.2rem !important;
        margin: 0;
    }
    .prediction-box p {
        color: #a5d6a7;
        margin: 4px 0 0 0;
        font-size: 1rem;
    }

    /* Info box */
    .info-box {
        background-color: #e8f5e9;
        border-left: 4px solid #2d6a4f;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #1a3c2e;
    }

    /* Warning box */
    .warn-box {
        background-color: #fff8e1;
        border-left: 4px solid #f9a825;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #5d4037;
    }

    /* Yield gap callout — NEW */
    .gap-callout {
        background: linear-gradient(135deg, #1a3c2e, #2d6a4f);
        border-radius: 10px;
        padding: 20px 24px;
        margin: 12px 0 20px 0;
        color: white;
    }
    .gap-callout h4 {
        color: #95d5b2 !important;
        margin: 0 0 8px 0;
        font-size: 1rem;
        font-family: Georgia, serif;
        letter-spacing: 0.5px;
    }
    .gap-callout p {
        color: #d8f3dc;
        margin: 0;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .gap-callout .gap-number {
        color: #52b788;
        font-size: 2rem;
        font-weight: bold;
        font-family: Georgia, serif;
    }

    /* Onboarding banner — NEW */
    .onboard-box {
        background-color: #e8f5e9;
        border: 1.5px solid #52b788;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
        font-size: 0.93rem;
        color: #1a3c2e;
        line-height: 1.7;
    }
    .onboard-box b { color: #2d6a4f; }

    /* Slider hint text — NEW */
    .slider-hint {
        font-size: 0.78rem;
        color: #a5d6a7;
        margin: -8px 0 10px 0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD DATA AND MODELS
# =============================================================================

BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    rf     = joblib.load(os.path.join(BASE, "rf_model_final.pkl"))
    svr    = joblib.load(os.path.join(BASE, "svr_model_final.pkl"))
    scaler = joblib.load(os.path.join(BASE, "scaler_final.pkl"))
    return rf, svr, scaler

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, "master_merged_final.csv"))

rf_model, svr_model, scaler = load_models()
master = load_data()

# Model feature order — must match training
MODEL_FEATURES = [
    'Rain_LongRains_Fraction',
    'Humidity_ShortRains_pct',
    'Rain_DrySeason_mm',
    'Temp_ShortRains_C',
    'SoilWetness_ShortRains',
    'Rain_OffSeason_mm',
]

LOO_RMSE      = 0.3640
COUNTY_MEAN   = master['Yield_Tonnes_Ha'].mean()
COUNTY_MAX    = master['Yield_Tonnes_Ha'].max()
COUNTY_MIN    = master['Yield_Tonnes_Ha'].min()
SEED_POTENTIAL = 9.5   # midpoint of 9–11 t/ha

PALETTE = {
    'green':      '#2d6a4f',
    'dark_green': '#1a3c2e',
    'light_green':'#52b788',
    'red':        '#e63946',
    'orange':     '#f4a261',
    'blue':       '#457b9d',
    'grey':       '#6c757d',
}

# =============================================================================
# PREDICTION HELPER
# =============================================================================

def make_prediction(rain_lr, rain_sr, rain_off, rain_dry,
                    temp_sr, humid_sr, wetness_sr):
    """Run both models and return predictions with uncertainty."""
    rain_annual  = rain_lr + rain_sr + rain_off + rain_dry
    lr_fraction  = rain_lr / rain_annual if rain_annual > 0 else 0.0

    row = pd.DataFrame([{
        'Rain_LongRains_Fraction': lr_fraction,
        'Humidity_ShortRains_pct': humid_sr,
        'Rain_DrySeason_mm':       rain_dry,
        'Temp_ShortRains_C':       temp_sr,
        'SoilWetness_ShortRains':  wetness_sr,
        'Rain_OffSeason_mm':       rain_off,
    }])[MODEL_FEATURES]

    scaled   = scaler.transform(row)
    rf_pred  = float(rf_model.predict(scaled)[0])
    svr_pred = float(svr_model.predict(scaled)[0])
    avg_pred = (rf_pred + svr_pred) / 2

    return {
        'rf':         round(rf_pred,  3),
        'svr':        round(svr_pred, 3),
        'avg':        round(avg_pred, 3),
        'low':        round(avg_pred - LOO_RMSE, 3),
        'high':       round(avg_pred + LOO_RMSE, 3),
        'lr_fraction':round(lr_fraction, 3),
        'annual_rain':round(rain_annual, 0),
    }

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("##  Maize Yield Predictor")
    st.markdown("**Uasin Gishu County, Kenya**")
    st.markdown("---")

    # ── ONBOARDING HINT IN SIDEBAR ────────────────────────────────────────────
    st.markdown("""
    <div style="background:#2d6a4f; border-radius:8px; padding:12px 14px;
                font-size:0.82rem; color:#d8f3dc; line-height:1.6; margin-bottom:12px;">
    <b style="color:#95d5b2;">How to use:</b><br>
    1. Adjust the sliders below to match your season's expected weather.<br>
    2. Click <b>Run Prediction</b>.<br>
    3. Read your yield estimate in the <b>Yield Predictor</b> tab.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Enter Seasonal Conditions")

    st.markdown("**Rainfall by Season (mm)**")

    rain_lr = st.slider("Long Rains — Mar to May", 100, 1200, 600, 10,
                         help="Rainfall during the main planting season (March–May). "
                              "This is the most critical window for maize growth.")
    st.markdown('<p class="slider-hint">Main planting season — most important for yield</p>',
                unsafe_allow_html=True)

    rain_sr = st.slider("Short Rains — Sep to Nov", 100, 900, 400, 10,
                         help="Rainfall in the short rains season (September–November). "
                              "Affects soil condition before the next planting.")
    st.markdown('<p class="slider-hint">Pre-planting period — shapes soil health</p>',
                unsafe_allow_html=True)

    rain_off = st.slider("Off Season — Jun to Aug", 50, 600, 150, 10,
                          help="Rainfall during the normally dry period (June–August). "
                               "Unexpected rain here can cause disease and soil problems.")
    st.markdown('<p class="slider-hint">Normally dry — unexpected rain causes problems</p>',
                unsafe_allow_html=True)

    rain_dry = st.slider("Dry Season — Dec to Feb", 20, 400, 80, 10,
                          help="Rainfall during land preparation (December–February). "
                               "Too much rain at this stage damages soil before planting.")
    st.markdown('<p class="slider-hint">Land preparation period — excess rain is harmful</p>',
                unsafe_allow_html=True)

    st.markdown("**Short Rains Conditions (Sep–Nov)**")

    temp_sr = st.slider("Average Temperature (°C)", 17.0, 23.0, 20.0, 0.1,
                         help="Average temperature during September–November.")
    st.markdown('<p class="slider-hint">Warmer = better soil recovery before planting</p>',
                unsafe_allow_html=True)

    humid_sr = st.slider("Average Humidity (%)", 55.0, 85.0, 70.0, 0.5,
                          help="Average humidity September–November. "
                               "High humidity (above ~75%) signals disease and waterlogging risk.")
    st.markdown('<p class="slider-hint">Above 75% = elevated disease risk</p>',
                unsafe_allow_html=True)

    wetness_sr = st.slider("Soil Wetness (0 – 1)", 0.3, 0.9, 0.6, 0.01,
                            help="How saturated the soil is on a 0–1 scale. "
                                 "0 = completely dry, 0.5 = moderately moist, 1 = fully waterlogged.")
    st.markdown('<p class="slider-hint">0 = bone dry · 0.5 = moist · 1 = waterlogged</p>',
                unsafe_allow_html=True)

    st.markdown("---")
    predict_btn = st.button("▶ Run Prediction", type="primary", use_container_width=True)

# Compute prediction on every slider change
prediction = make_prediction(rain_lr, rain_sr, rain_off, rain_dry,
                              temp_sr, humid_sr, wetness_sr)

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
#  Maize Yield Prediction Dashboard
### Uasin Gishu County, Kenya &nbsp;|&nbsp; 2012 – 2023 &nbsp;|&nbsp; KCA Tech Expo · March 2026
""")

# ── ONBOARDING BANNER (main area) ─────────────────────────────────────────────
st.markdown("""
<div class="onboard-box">
<b>Welcome to the Maize Yield Prediction Dashboard.</b>
This tool uses 12 years of historical weather, soil, and harvest data from Uasin Gishu County
to estimate how much maize a season is likely to produce — measured in
<b>tonnes per hectare (t/ha)</b>, which means tonnes of grain harvested from every hectare of land.<br><br>
<b>To get a yield prediction:</b> Use the sliders in the left panel to enter your season's
expected rainfall and temperature conditions, then click the
<b>Yield Predictor</b> tab above. No technical knowledge is required.
The four tabs below cover: overall trends, detailed data, the live prediction tool, and how accurate the model is.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "  Overview  ",
    "  Data Exploration  ",
    "  Yield Predictor  ",
    "  Model Accuracy  ",
])

# =============================================================================
# TAB 1 — OVERVIEW
# =============================================================================

with tab1:

    # ── YIELD GAP CALLOUT — NEW ───────────────────────────────────────────────
    st.markdown("""
    <div class="gap-callout">
        <h4>⚠ THE MOST IMPORTANT FINDING: The Yield Gap is a Soil Problem, Not a Weather Problem</h4>
        <p>
        Farmers in Uasin Gishu County harvest an average of
        <span class="gap-number">3.65 t/ha</span>
        — but the seeds they plant are capable of producing
        <span class="gap-number">9–11 t/ha</span>.
        That is a gap of roughly <b style="color:#52b788;">6.5 tonnes per hectare</b> of lost potential every season.
        <br><br>
        <b style="color:#95d5b2;">Weather improvements alone cannot close this gap.</b>
        Even in 2018 — the best weather year on record — yield only reached 4.26 t/ha.
        The root cause is <b style="color:#52b788;">soil acidity: pH 5.7</b> (the safe range for maize is 6.0–7.0).
        At pH 5.7, aluminium in the soil becomes toxic to maize roots and blocks nutrient uptake —
        even when fertiliser is applied at full dose.
        <b style="color:#95d5b2;">Agricultural lime applied at 1–2 tonnes per hectare</b>
        would raise the pH and is the single highest-impact intervention available.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("County Average Yield",   f"{COUNTY_MEAN:.2f} t/ha",  "2012 – 2023")
    c2.metric("Best Year on Record",    f"{COUNTY_MAX:.2f} t/ha",   "2018 — H6213 variety")
    c3.metric("Worst Year on Record",   f"{COUNTY_MIN:.2f} t/ha",   "2012 — MLN outbreak")
    c4.metric("Yield Gap vs Potential", "~6.5 t/ha",                "Seed potential: 9–11 t/ha")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # ── YIELD TREND CHART with seed-potential benchmark line — IMPROVED ──
        fig = go.Figure()

        # Shaded fill under actual yield
        fig.add_trace(go.Scatter(
            x=master['Year'], y=master['Yield_Tonnes_Ha'],
            mode='lines+markers',
            line=dict(color=PALETTE['green'], width=2.5),
            marker=dict(size=9, color=PALETTE['green'],
                        line=dict(color='white', width=2)),
            fill='tozeroy', fillcolor='rgba(45,106,79,0.08)',
            name='Actual Yield',
            hovertemplate='<b>%{x}</b><br>Yield: %{y:.3f} t/ha<extra></extra>',
        ))

        # County mean dashed line
        fig.add_hline(
            y=COUNTY_MEAN,
            line_dash='dash',
            line_color=PALETTE['grey'],
            line_width=1.5,
            annotation_text=f"County average: {COUNTY_MEAN:.2f} t/ha",
            annotation_position="bottom right",
        )

        # Seed potential benchmark line — NEW
        fig.add_hline(
            y=SEED_POTENTIAL,
            line_dash='dot',
            line_color=PALETTE['orange'],
            line_width=2,
            annotation_text="Seed potential: ~9.5 t/ha  ← What seeds CAN produce",
            annotation_position="top right",
            annotation_font_color=PALETTE['orange'],
        )

        # Yield gap shaded band between actual mean and seed potential — NEW
        fig.add_hrect(
            y0=COUNTY_MEAN,
            y1=SEED_POTENTIAL,
            fillcolor="rgba(244,162,97,0.08)",
            line_width=0,
            annotation_text="Yield gap (~6.5 t/ha)",
            annotation_position="top left",
            annotation_font_color=PALETTE['orange'],
            annotation_font_size=11,
        )

        fig.update_layout(
            title='Annual Maize Yield — Uasin Gishu County<br>'
                  '<sup>The gap between the orange dotted line and the actual yield '
                  'represents lost harvest potential every single season</sup>',
            xaxis_title='Year',
            yaxis_title='Yield (tonnes per hectare)',
            xaxis=dict(tickmode='array', tickvals=master['Year'].tolist()),
            yaxis=dict(range=[0, 11.5]),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=380,
            margin=dict(t=70, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

    with col_right:
        st.markdown("### Key Findings")
        st.markdown("""
        <div class="info-box">
        <b>Rainfall timing matters more than total volume.</b><br>
        In 2020, a record 2,371 mm of rain fell — yet yield was only 3.07 t/ha,
        one of the worst seasons on record. Only 34% of that rain arrived during
        the planting season. Rain at the wrong time does not help the crop.
        </div>

        <div class="info-box">
        <b> Conditions before planting shape the harvest.</b><br>
        Humidity and soil moisture in September–November predict the
        following year's yield more strongly than the planting season
        itself. The harvest is decided months before planting begins.
        </div>

        <div class="warn-box">
        <b> The yield gap is a soil problem.</b><br>
        Average yield is 3.65 t/ha against seed potential of 9–11 t/ha.
        Soil acidity (pH 5.7) blocks nutrient uptake. Liming raises pH
        and unlocks what better weather alone cannot achieve.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Year-by-Year Summary")
    display_cols = ['Year', 'Yield_Tonnes_Ha', 'Rain_LongRains_mm',
                    'Rain_Annual_mm', 'Temp_LongRains_C', 'Variety', 'MLN_Risk']
    display_cols = [c for c in display_cols if c in master.columns]
    display_df = master[display_cols].copy().rename(columns={
        'Yield_Tonnes_Ha':   'Yield (t/ha)',
        'Rain_LongRains_mm': 'Long Rains (mm)',
        'Rain_Annual_mm':    'Annual Rain (mm)',
        'Temp_LongRains_C':  'Temp Long Rains (°C)',
        'MLN_Risk':          'MLN Risk',
    })
    st.dataframe(
        display_df.style.format({
            'Yield (t/ha)':         '{:.3f}',
            'Long Rains (mm)':      '{:.0f}',
            'Annual Rain (mm)':     '{:.0f}',
            'Temp Long Rains (°C)': '{:.2f}',
        }).background_gradient(subset=['Yield (t/ha)'], cmap='Greens'),
        use_container_width=True,
        hide_index=True,
    )

# =============================================================================
# TAB 2 — DATA EXPLORATION
# =============================================================================

with tab2:

    # ── ORIENTATION NOTE FOR NON-TECHNICAL USERS — NEW ───────────────────────
    st.markdown("""
    <div class="info-box">
    <b>This tab is for detailed data exploration.</b>
    If you just want a yield prediction, go straight to the <b> Yield Predictor</b> tab.<br>
    Here you can see how each weather variable is related to yield,
    and explore scatter plots for any variable you choose.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How Each Variable Relates to Yield")
    st.markdown("""
    <div class="info-box">
    The chart below shows the <b>correlation</b> between each weather variable and yield.
    Think of correlation as a score from −1 to +1:<br>
    &nbsp;&nbsp;• <b>Closer to +1</b> = when this goes up, yield tends to go up<br>
    &nbsp;&nbsp;• <b>Closer to −1</b> = when this goes up, yield tends to go down<br>
    &nbsp;&nbsp;• <b>Near 0</b> = little or no relationship with yield<br>
    The dashed lines mark the ±0.5 threshold — variables beyond those lines
    are considered <b>strong predictors</b>.
    </div>
    """, unsafe_allow_html=True)

    # Correlation bar chart
    EXCLUDE = ['Year', 'Soil_pH', 'Nitrogen_g_kg', 'Fertilizer_Kg_Ha',
               'Yield_Tonnes_Ha', 'Yield_Potential_Min_tHa',
               'Yield_Potential_Max_tHa', 'Area_Planted_Ha', 'Variety_Code',
               'MLN_Risk']
    num_cols  = master.select_dtypes(include='number').columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE]
    corr      = master[feat_cols + ['Yield_Tonnes_Ha']].corr()['Yield_Tonnes_Ha'].drop('Yield_Tonnes_Ha')
    corr_df   = pd.DataFrame({'Feature': corr.index, 'Correlation': corr.values})
    corr_df   = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=True).index)

    colors = [PALETTE['green'] if v > 0 else PALETTE['red']
              for v in corr_df['Correlation']]

    fig_corr = go.Figure(go.Bar(
        x=corr_df['Correlation'],
        y=corr_df['Feature'],
        orientation='h',
        marker_color=colors,
        marker_line_color='white',
        marker_line_width=0.5,
        hovertemplate='<b>%{y}</b><br>r = %{x:.4f}<extra></extra>',
    ))
    fig_corr.add_vline(x=0,    line_color='black',          line_width=1)
    fig_corr.add_vline(x=0.5,  line_color=PALETTE['green'], line_width=1.2,
                       line_dash='dash')
    fig_corr.add_vline(x=-0.5, line_color=PALETTE['red'],   line_width=1.2,
                       line_dash='dash')
    fig_corr.update_layout(
        title='How Strongly Each Variable is Related to Yield<br>'
              '<sup>Green bars = higher value leads to better yield  |  '
              'Red bars = higher value leads to lower yield  |  '
              'Dashed lines = strong predictor threshold (±0.5)</sup>',
        xaxis_title='Relationship strength (−1 to +1)',
        xaxis_range=[-1.05, 1.05],
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=520,
        margin=dict(t=70, l=250),
    )
    st.plotly_chart(fig_corr, use_container_width=True, config={"responsive": True})

    st.markdown("---")
    st.markdown("### Explore Any Variable vs Yield")

    scatter_options = [c for c in feat_cols if c in master.columns]
    selected = st.selectbox(
        "Choose a weather variable to compare against yield:",
        scatter_options,
        index=scatter_options.index('Rain_LongRains_Fraction')
        if 'Rain_LongRains_Fraction' in scatter_options else 0
    )

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        r_val = master['Yield_Tonnes_Ha'].corr(master[selected])
        z     = np.polyfit(master[selected], master['Yield_Tonnes_Ha'], 1)
        x_min, x_max = master[selected].min(), master[selected].max()
        x_line = np.linspace(x_min, x_max, 100)

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=x_line, y=np.poly1d(z)(x_line),
            mode='lines',
            line=dict(color=PALETTE['grey'], dash='dash', width=1.5),
            name='Trend',
            showlegend=False,
        ))
        fig_sc.add_trace(go.Scatter(
            x=master[selected], y=master['Yield_Tonnes_Ha'],
            mode='markers+text',
            marker=dict(size=11, color=PALETTE['blue'],
                        line=dict(color='white', width=1.5)),
            text=master['Year'].astype(str),
            textposition='top right',
            textfont=dict(size=9),
            name='Year',
            hovertemplate='<b>%{text}</b><br>'
                          f'{selected}: %{{x:.2f}}<br>'
                          'Yield: %{y:.3f} t/ha<extra></extra>',
        ))
        fig_sc.update_layout(
            title=f'{selected} vs Yield — relationship strength: {r_val:+.3f}',
            xaxis_title=selected,
            yaxis_title='Yield (tonnes per hectare)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=380,
        )
        st.plotly_chart(fig_sc, use_container_width=True, config={"responsive": True})

    with col_s2:
        st.markdown("#### What This Means")
        strength  = "Strong" if abs(r_val) > 0.5 else ("Moderate" if abs(r_val) > 0.3 else "Weak")
        direction = "positive" if r_val > 0 else "negative"

        plain_direction = (
            "When this variable is higher, yield tends to be higher."
            if r_val > 0 else
            "When this variable is higher, yield tends to be lower."
        )

        st.markdown(f"""
        <div class="info-box">
        <b>Relationship strength: {r_val:+.4f}</b><br>
        This is a <b>{strength.lower()} {direction}</b> relationship.<br><br>
        {plain_direction}
        </div>
        """, unsafe_allow_html=True)

        if abs(r_val) < 0.3:
            st.markdown("""
            <div class="warn-box">
            <b>Weak relationship</b> — this variable alone is not a reliable
            predictor. The model combines multiple weaker signals together
            to improve overall accuracy.
            </div>
            """, unsafe_allow_html=True)
        elif abs(r_val) >= 0.5:
            st.markdown("""
            <div class="info-box">
            <b>Strong predictor</b> — this variable on its own explains
            a meaningful share of the year-to-year variation in yield.
            It was likely selected as one of the 6 model inputs.
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# TAB 3 — YIELD PREDICTOR
# =============================================================================

with tab3:

    st.markdown("### Interactive Yield Predictor")
    st.markdown("""
    <div class="info-box">
    <b>How this works:</b> Adjust the sliders in the left panel to match your season's
    expected weather. The prediction updates automatically. The result is shown in
    <b>tonnes per hectare (t/ha)</b> — for example, 4.0 t/ha means
    4 tonnes of maize harvested from every hectare of land.
    The uncertainty range (±0.36 t/ha) means the true yield will typically
    fall within that band around the predicted number.
    </div>
    """, unsafe_allow_html=True)

    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        pred_val  = prediction['avg']
        pred_low  = prediction['low']
        pred_high = prediction['high']

        if pred_val >= COUNTY_MEAN + 0.2:
            verdict      = "✅ Above average season"
            verdict_sub  = f"Better than the county average of {COUNTY_MEAN:.2f} t/ha"
            v_color      = "#2d6a4f"
        elif pred_val <= COUNTY_MEAN - 0.2:
            verdict      = "⚠ Below average season"
            verdict_sub  = f"Below the county average of {COUNTY_MEAN:.2f} t/ha"
            v_color      = "#e63946"
        else:
            verdict      = "➡ Average season"
            verdict_sub  = f"Close to the county average of {COUNTY_MEAN:.2f} t/ha"
            v_color      = "#f4a261"

        st.markdown(f"""
        <div class="prediction-box">
            <p>Predicted Maize Yield</p>
            <h1>{pred_val:.2f} t/ha</h1>
            <p>Likely range: {pred_low:.2f} – {pred_high:.2f} t/ha</p>
            <p style="color:white; font-size:0.95rem; margin-top:10px; font-weight:bold;">
                {verdict}
            </p>
            <p style="color:#a5d6a7; font-size:0.82rem; margin-top:2px;">
                {verdict_sub}
            </p>
            <p style="color:#d8f3dc; font-size:0.78rem; margin-top:10px; opacity:0.75;">
                Model A (Random Forest): {prediction['rf']:.2f} t/ha &nbsp;|&nbsp;
                Model B (SVR): {prediction['svr']:.2f} t/ha
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_val,
            delta={
                'reference':    COUNTY_MEAN,
                'valueformat':  '.2f',
                'suffix':       ' t/ha vs county average',
            },
            gauge={
                'axis': {
                    'range':       [2.0, 5.0],
                    'ticksuffix':  ' t/ha',
                    'tickfont':    {'size': 10},
                },
                'bar': {'color': PALETTE['green']},
                'steps': [
                    {'range': [2.0, 3.0], 'color': '#ffebee'},
                    {'range': [3.0, 3.5], 'color': '#fff3e0'},
                    {'range': [3.5, 4.0], 'color': '#e8f5e9'},
                    {'range': [4.0, 5.0], 'color': '#c8e6c9'},
                ],
                'threshold': {
                    'line':      {'color': PALETTE['grey'], 'width': 2},
                    'thickness': 0.75,
                    'value':     COUNTY_MEAN,
                },
            },
            title={
                'text': (
                    f"Yield Prediction<br>"
                    f"<span style='font-size:0.75em; color:{v_color}'>{verdict}</span>"
                )
            },
            number={'suffix': ' t/ha', 'valueformat': '.2f'},
        ))
        fig_gauge.update_layout(
            height=290,
            paper_bgcolor='white',
            margin=dict(t=50, b=10),
            annotations=[dict(
                x=0.5, y=-0.08,
                text=(
                    "<span style='font-size:11px; color:#888;'>"
                    "Red zone = poor season &nbsp;|&nbsp; "
                    "Amber = below average &nbsp;|&nbsp; "
                    "Green zones = good season"
                    "</span>"
                ),
                showarrow=False,
                xref='paper', yref='paper',
                font=dict(size=10),
            )]
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"responsive": True})

    with col_p2:
        st.markdown("#### Your Input Summary")

        rain_annual = rain_lr + rain_sr + rain_off + rain_dry
        lr_fraction = rain_lr / rain_annual if rain_annual > 0 else 0

        timing_label = (
            "✅ Good — enough rain arrived during the planting window"
            if lr_fraction > 0.35 else
            "⚠ Poor — most rain fell outside the planting season"
        )
        timing_color = PALETTE['green'] if lr_fraction > 0.35 else PALETTE['red']

        # Rainfall breakdown donut
        fig_donut = go.Figure(go.Pie(
            labels=['Long Rains (Mar–May)', 'Short Rains (Sep–Nov)',
                    'Off Season (Jun–Aug)', 'Dry Season (Dec–Feb)'],
            values=[rain_lr, rain_sr, rain_off, rain_dry],
            hole=0.55,
            marker_colors=[
                PALETTE['green'], PALETTE['light_green'],
                PALETTE['orange'], PALETTE['blue'],
            ],
            textinfo='label+percent',
            textfont_size=10,
            hovertemplate='<b>%{label}</b><br>%{value:.0f} mm (%{percent})<extra></extra>',
        ))
        fig_donut.update_layout(
            title=f'How your {rain_annual:.0f} mm of rain is distributed across seasons',
            height=290,
            paper_bgcolor='white',
            margin=dict(t=55, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"responsive": True})

        st.markdown(f"""
        <div class="info-box">
        <b>Planting-season rain share:</b> {lr_fraction:.1%}<br>
        <span style="color:{timing_color}">{timing_label}</span><br>
        <span style="font-size:0.82rem; color:#555;">
        Seasons above 40% planting-season share consistently produce above-average yields.
        </span>
        </div>

        <div class="info-box">
        <b>Total annual rainfall:</b> {rain_annual:.0f} mm<br>
        Historical range in Uasin Gishu: 1,142 – 2,371 mm
        </div>

        <div class="warn-box">
        <b>Prediction uncertainty:</b> The model is accurate to within
        ±0.36 t/ha on average — meaning the true yield will usually
        fall within the range shown above.
        This uncertainty comes from testing the model against
        12 years of real historical data.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Compare Different Scenarios")
    st.markdown("""
    <div class="info-box">
    <b>How to use this tool:</b> Set the sliders to one set of conditions and click
    <b>Save Current Scenario</b>. Then change the sliders to a different scenario
    and save again. The chart below will compare your saved scenarios side by side.<br>
    <i>Example: compare a good Long Rains year (600mm, 50%+ in planting window)
    against a poor year (200mm, 25% in planting window).</i>
    </div>
    """, unsafe_allow_html=True)

    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = []

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button(" Save Current Scenario"):
            st.session_state.scenarios.append({
                'Scenario':         f"Scenario {len(st.session_state.scenarios)+1}",
                'Long Rains (mm)':  rain_lr,
                'Annual Rain (mm)': rain_annual,
                'LR Share':         f"{lr_fraction:.1%}",
                'Prediction (t/ha)':pred_val,
                'Likely Range':     f"{pred_low:.2f} – {pred_high:.2f}",
            })
    with col_btn2:
        if st.button(" Clear All Scenarios"):
            st.session_state.scenarios = []

    if st.session_state.scenarios:
        sc_df = pd.DataFrame(st.session_state.scenarios)
        fig_sc_bar = px.bar(
            sc_df, x='Scenario', y='Prediction (t/ha)',
            color='Prediction (t/ha)',
            color_continuous_scale=['#e63946', '#f4a261', '#2d6a4f'],
            range_color=[2.5, 4.5],
            text='Prediction (t/ha)',
        )
        fig_sc_bar.add_hline(
            y=COUNTY_MEAN,
            line_dash='dash',
            line_color=PALETTE['grey'],
            annotation_text=f"County average: {COUNTY_MEAN:.2f} t/ha",
        )
        fig_sc_bar.update_traces(
            texttemplate='%{text:.2f} t/ha',
            textposition='outside',
        )
        fig_sc_bar.update_layout(
            title='Scenario Comparison — which set of conditions produces the better harvest?',
            yaxis_title='Predicted Yield (tonnes per hectare)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            coloraxis_showscale=False,
            height=340,
        )
        st.plotly_chart(fig_sc_bar, use_container_width=True, config={"responsive": True})
        st.dataframe(sc_df, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div class="warn-box">
        No scenarios saved yet. Adjust the sliders and click
        <b> Save Current Scenario</b> to start comparing.
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 4 — MODEL ACCURACY (renamed from "Model Performance")
# =============================================================================

with tab4:

    st.markdown("### How Accurate Is the Model?")

    # ── PLAIN-ENGLISH EXPLANATION — IMPROVED ─────────────────────────────────
    st.markdown("""
    <div class="info-box">
    <b>How was the model tested?</b> With only 12 years of data, the model was validated
    using a method called <b>Leave-One-Out</b> testing — which works like this:
    hold out one year, train the model on the remaining 11 years, then predict
    the held-out year. Repeat for all 12 years. The result is 12 genuine predictions,
    each made on data the model had never seen. <b>Every accuracy figure below comes
    from this honest test — not from data the model was trained on.</b>
    </div>
    """, unsafe_allow_html=True)

    # ── METRIC CARDS — PLAIN ENGLISH LABELS ──────────────────────────────────
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric(
        "Average Prediction Error",
        "0.364 t/ha",
        "Random Forest model",
        help="On average, the model's prediction is within 0.364 t/ha of the actual harvest. "
             "That is about 10% of the county's typical yield.",
    )
    col_m2.metric(
        "How Much Variation Explained",
        "30%",
        "Weather explains ~30% of year-to-year change",
        help="30% of the year-to-year variation in yield can be predicted from weather alone. "
             "The remaining 70% is driven by factors like soil pH, disease, and seed variety.",
    )
    col_m3.metric(
        "Error as % of Typical Yield",
        "10%",
        "County average is 3.65 t/ha",
        help="The average error of 0.364 t/ha is about 10% of the county's average yield of 3.65 t/ha. "
             "In practical terms: if you expect 4 t/ha, the real harvest will typically fall between 3.6 and 4.4 t/ha.",
    )
    col_m4.metric(
        "Model Bias",
        "−0.007 t/ha",
        "Essentially unbiased",
        help="A bias near zero means the model does not consistently over-predict or under-predict. "
             "It makes errors in both directions roughly equally.",
    )

    st.markdown("---")

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        loo_preds = np.array([3.550, 3.813, 3.756, 3.750, 3.891,
                               3.422, 3.530, 3.547, 3.451, 3.804,
                               3.495, 3.690])
        actual = master['Yield_Tonnes_Ha'].values
        years  = master['Year'].values

        fig_avp = go.Figure()
        lo = min(actual.min(), loo_preds.min()) - 0.25
        hi = max(actual.max(), loo_preds.max()) + 0.25

        fig_avp.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode='lines',
            line=dict(color=PALETTE['grey'], dash='dash', width=1.5),
            name='Perfect prediction',
            showlegend=True,
        ))
        # Error band
        fig_avp.add_trace(go.Scatter(
            x=[lo, hi], y=[lo + LOO_RMSE, hi + LOO_RMSE],
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip',
        ))
        fig_avp.add_trace(go.Scatter(
            x=[lo, hi], y=[lo - LOO_RMSE, hi - LOO_RMSE],
            mode='lines', fill='tonexty',
            fillcolor='rgba(45,106,79,0.08)',
            line=dict(width=0),
            name=f'±{LOO_RMSE:.2f} t/ha acceptable error band',
        ))
        fig_avp.add_trace(go.Scatter(
            x=actual, y=loo_preds,
            mode='markers+text',
            marker=dict(size=12, color=PALETTE['green'],
                        line=dict(color='white', width=2)),
            text=[str(y) for y in years],
            textposition='top right',
            textfont=dict(size=9),
            name='Predicted vs actual',
            hovertemplate='<b>%{text}</b><br>Actual: %{x:.3f} t/ha<br>'
                          'Predicted: %{y:.3f} t/ha<extra></extra>',
        ))
        fig_avp.update_layout(
            title='Predicted vs Actual Yield for Every Year<br>'
                  '<sup>Points close to the dashed line = accurate predictions. '
                  'Shaded band = acceptable error range.</sup>',
            xaxis_title='Actual Yield (t/ha)',
            yaxis_title='What the Model Predicted (t/ha)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
            xaxis=dict(range=[lo, hi]),
            yaxis=dict(range=[lo, hi]),
        )
        st.plotly_chart(fig_avp, use_container_width=True, config={"responsive": True})

    with col_v2:
        # ── RESIDUALS CHART — NEUTRAL COLOURS ────────────────────────────────
        residuals  = loo_preds - actual
        # Neutral palette: teal for over-estimate, coral for under-estimate
        res_colors = ['#457b9d' if r >= 0 else '#e07a5f' for r in residuals]
        res_labels = ['Model slightly high' if r >= 0 else 'Model slightly low'
                      for r in residuals]

        fig_res = go.Figure(go.Bar(
            x=years,
            y=residuals,
            marker_color=res_colors,
            marker_line_color='white',
            text=[f'{r:+.2f}' for r in residuals],
            textposition='outside',
            customdata=res_labels,
            hovertemplate=(
                '<b>%{x}</b><br>'
                'Error: %{y:+.3f} t/ha<br>'
                '%{customdata}<extra></extra>'
            ),
        ))
        fig_res.add_hline(y=0,          line_color='black',       line_width=1.2)
        fig_res.add_hline(y=LOO_RMSE,   line_color=PALETTE['grey'],
                          line_dash='dash', line_width=1.2,
                          annotation_text=f'Avg error: +{LOO_RMSE:.2f}')
        fig_res.add_hline(y=-LOO_RMSE,  line_color=PALETTE['grey'],
                          line_dash='dash', line_width=1.2,
                          annotation_text=f'Avg error: −{LOO_RMSE:.2f}')
        fig_res.update_layout(
            title='How Far Off Was the Model Each Year?<br>'
                  '<sup>Blue = model predicted slightly too high  |  '
                  'Coral = model predicted slightly too low  |  '
                  'Dashed lines = average error boundary</sup>',
            xaxis_title='Year',
            yaxis_title='Prediction error (t/ha)',
            xaxis=dict(tickmode='array', tickvals=years.tolist()),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
        )
        st.plotly_chart(fig_res, use_container_width=True, config={"responsive": True})

    st.markdown("---")

    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        # ── FEATURE IMPORTANCE — PLAIN ENGLISH ───────────────────────────────
        perm_importance = {
            'Humidity — Short Rains (Sep–Nov)': 0.0290,
            'Rainfall — Dry Season (Dec–Feb)':  0.0287,
            'Long Rains Rainfall Share':         0.0249,
            'Temperature — Short Rains':         0.0132,
            'Rainfall — Off Season (Jun–Aug)':   0.0073,
            'Soil Wetness — Short Rains':        0.0041,
        }
        imp_df = pd.DataFrame(
            list(perm_importance.items()),
            columns=['Variable', 'Importance']
        ).sort_values('Importance')

        fig_imp = go.Figure(go.Bar(
            x=imp_df['Importance'],
            y=imp_df['Variable'],
            orientation='h',
            marker_color=PALETTE['green'],
            marker_line_color='white',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
        ))
        fig_imp.update_layout(
            title='Which Variables Matter Most to the Model?<br>'
                  '<sup>Longer bar = removing this variable makes predictions less accurate</sup>',
            xaxis_title='How much accuracy is lost without this variable',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=340,
            margin=dict(l=230),
        )
        st.plotly_chart(fig_imp, use_container_width=True, config={"responsive": True})

        st.markdown("""
        <div class="info-box">
        <b>Key insight:</b> 3 of the top 4 most important variables describe
        conditions in September–February — <i>before</i> the main planting season begins.
        The harvest outcome is largely determined before the farmer even plants.
        </div>
        """, unsafe_allow_html=True)

    with col_f2:
        # ── RELIABILITY TABLE — PLAIN ENGLISH LABELS ─────────────────────────
        st.markdown("### Overall Model Reliability")
        st.markdown("""
        <div class="info-box">
        The table below rates different aspects of how the model was built.
        Five stars means excellent; fewer stars means there is room for improvement.
        </div>
        """, unsafe_allow_html=True)

        reliability = {
            'Is the seasonal design sound?':           ('★★★★★', 'Year split into 4 agronomic seasons — captures timing, not just totals'),
            'Is the accuracy test honest?':            ('★★★★★', 'Leave-One-Out testing — no year was tested on data it trained with'),
            'Is the model free of hidden errors?':     ('★★★★★', 'Yield variable excluded from inputs — no data leakage'),
            'Are the predictors well chosen?':         ('★★★★☆', '6 variables from 38 candidates — justified by both data and crop science'),
            'Is the model algorithm appropriate?':     ('★★★★☆', 'Random Forest and SVR are well suited to small datasets'),
            'Can results be trusted for planning?':    ('★★★☆☆', 'Directional patterns are reliable — precise numbers should be treated as estimates'),
            'Is there enough historical data?':        ('★★☆☆☆', 'Only 12 years — extending to 20+ years would improve all estimates'),
        }
        rel_df = pd.DataFrame(
            [(k, v[0], v[1]) for k, v in reliability.items()],
            columns=['Question', 'Rating', 'What it means']
        )
        st.dataframe(rel_df, use_container_width=True, hide_index=True, height=300)

        st.markdown("""
        <div class="warn-box">
        <b>Bottom line:</b> This model reliably identifies which weather conditions
        lead to better or worse harvests, and is useful for seasonal planning
        and scenario comparison. It is not designed for precise yield forecasting
        until more historical data becomes available.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Full Results Table — All 12 Years")

    loo_preds_check = np.array([3.550, 3.813, 3.756, 3.750, 3.891,
                                 3.422, 3.530, 3.547, 3.451, 3.804,
                                 3.495, 3.690])
    residuals_check = loo_preds_check - actual
    pred_table = pd.DataFrame({
        'Year':               years,
        'Actual Yield (t/ha)':actual.round(3),
        'Model Prediction':   loo_preds_check.round(3),
        'Difference (t/ha)':  residuals_check.round(3),
        'Variety':            master['Variety'].values,
    })
    st.dataframe(
        pred_table.style.format({
            'Actual Yield (t/ha)': '{:.3f}',
            'Model Prediction':    '{:.3f}',
            'Difference (t/ha)':   '{:+.3f}',
        }).background_gradient(subset=['Actual Yield (t/ha)'], cmap='Greens'),
        use_container_width=True,
        hide_index=True,
    )

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding:8px 0;">
    Maize Yield Prediction — Uasin Gishu County, Kenya &nbsp;|&nbsp;
    KCA Tech Expo · March 2026 &nbsp;|&nbsp;
    Data: Ministry of Agriculture · NASA POWER · CIMMYT · Purdue University
</div>
""", unsafe_allow_html=True)
