# =============================================================================
# MAIZE YIELD PREDICTION DASHBOARD
# Uasin Gishu County, Kenya — IBM SkillsBuild Bootcamp
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
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD DATA AND MODELS
# =============================================================================

BASE = os.path.expanduser("~/Desktop/Maize_Yield_Submission")

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

LOO_RMSE     = 0.3640
COUNTY_MEAN  = master['Yield_Tonnes_Ha'].mean()
COUNTY_MAX   = master['Yield_Tonnes_Ha'].max()
COUNTY_MIN   = master['Yield_Tonnes_Ha'].min()

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
    st.markdown("## Maize Yield Predictor")
    st.markdown("**Uasin Gishu County, Kenya**")
    st.markdown("---")
    st.markdown("### Enter Seasonal Conditions")
    st.markdown("*Adjust the sliders to match your season's weather.*")

    st.markdown("**Rainfall by Season (mm)**")
    rain_lr  = st.slider("Long Rains — Mar to May",    100, 1200, 600, 10,
                          help="Rainfall during the main planting season")
    rain_sr  = st.slider("Short Rains — Sep to Nov",   100,  900, 400, 10,
                          help="Rainfall during the second season")
    rain_off = st.slider("Off Season — Jun to Aug",     50,  600, 150, 10,
                          help="Rainfall during the dry period")
    rain_dry = st.slider("Dry Season — Dec to Feb",     20,  400,  80, 10,
                          help="Rainfall during land preparation period")

    st.markdown("**Short Rains Conditions (Sep–Nov)**")
    temp_sr    = st.slider("Average Temperature (°C)",  17.0, 23.0, 20.0, 0.1)
    humid_sr   = st.slider("Average Humidity (%)",      55.0, 85.0, 70.0, 0.5)
    wetness_sr = st.slider("Soil Wetness (0 – 1)",       0.3,  0.9,  0.6, 0.01)

    st.markdown("---")
    predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

# Compute prediction on every slider change
prediction = make_prediction(rain_lr, rain_sr, rain_off, rain_dry,
                              temp_sr, humid_sr, wetness_sr)

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
# Maize Yield Prediction Dashboard
### Uasin Gishu County, Kenya &nbsp;|&nbsp; 2012 – 2023 &nbsp;|&nbsp; IBM SkillsBuild Bootcamp
""")
st.markdown("---")

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "  Overview  ",
    "  Data Exploration  ",
    "  Yield Predictor  ",
    "  Model Performance  ",
])

# =============================================================================
# TAB 1 — OVERVIEW
# =============================================================================

with tab1:

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("County Average Yield",  f"{COUNTY_MEAN:.2f} t/ha",  "2012 – 2023")
    c2.metric("Best Year on Record",   f"{COUNTY_MAX:.2f} t/ha",   "2018 — H6213 variety")
    c3.metric("Worst Year on Record",  f"{COUNTY_MIN:.2f} t/ha",   "2012 — MLN outbreak")
    c4.metric("Yield Gap vs Potential","~6.5 t/ha",                 "Seed potential: 9–11 t/ha")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Yield trend chart
        fig = go.Figure()
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
        fig.add_hline(y=COUNTY_MEAN, line_dash='dash',
                      line_color=PALETTE['grey'], line_width=1.5,
                      annotation_text=f"Mean: {COUNTY_MEAN:.2f} t/ha",
                      annotation_position="bottom right")
        fig.update_layout(
            title='Annual Maize Yield — Uasin Gishu County',
            xaxis_title='Year', yaxis_title='Yield (t/ha)',
            xaxis=dict(tickmode='array', tickvals=master['Year'].tolist()),
            plot_bgcolor='white', paper_bgcolor='white',
            height=340, margin=dict(t=50, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### Key Findings")
        st.markdown("""
        <div class="info-box">
        <b>Rainfall timing</b> matters more than total volume.
        2020 had record rainfall (2,371mm) but below-average yield
        because only 34% fell during the planting season.
        </div>
        <div class="info-box">
        <b>Pre-planting conditions</b> shape the harvest. Humidity
        and soil moisture in Sep–Nov predict the following year's
        yield more strongly than planting-season weather alone.
        </div>
        <div class="warn-box">
        <b>Structural yield gap.</b> Average yield is 3.6 t/ha
        against seed potential of 9–11 t/ha. The primary cause is
        soil acidity (pH 5.7). Liming would unlock what better
        weather cannot.
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
            'Yield (t/ha)': '{:.3f}',
            'Long Rains (mm)': '{:.0f}',
            'Annual Rain (mm)': '{:.0f}',
            'Temp Long Rains (°C)': '{:.2f}',
        }).background_gradient(subset=['Yield (t/ha)'], cmap='Greens'),
        use_container_width=True, hide_index=True,
    )

# =============================================================================
# TAB 2 — DATA EXPLORATION
# =============================================================================

with tab2:

    st.markdown("### How Each Variable Relates to Yield")

    # Correlation bar chart
    EXCLUDE = ['Year', 'Soil_pH', 'Nitrogen_g_kg', 'Fertilizer_Kg_Ha',
               'Yield_Tonnes_Ha', 'Yield_Potential_Min_tHa',
               'Yield_Potential_Max_tHa', 'Area_Planted_Ha', 'Variety_Code',
               'MLN_Risk']
    num_cols = master.select_dtypes(include='number').columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE]
    corr = master[feat_cols + ['Yield_Tonnes_Ha']].corr()['Yield_Tonnes_Ha'].drop('Yield_Tonnes_Ha')
    corr_df = pd.DataFrame({'Feature': corr.index, 'Correlation': corr.values})
    corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=True).index)

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
    fig_corr.add_vline(x=0,    line_color='black',           line_width=1)
    fig_corr.add_vline(x=0.5,  line_color=PALETTE['green'],  line_width=1.2,
                       line_dash='dash')
    fig_corr.add_vline(x=-0.5, line_color=PALETTE['red'],    line_width=1.2,
                       line_dash='dash')
    fig_corr.update_layout(
        title='Feature Correlations with Maize Yield<br>'
              '<sup>Green = higher value → better yield &nbsp;|&nbsp; '
              'Red = higher value → lower yield</sup>',
        xaxis_title='Pearson Correlation (r)',
        xaxis_range=[-1.05, 1.05],
        plot_bgcolor='white', paper_bgcolor='white',
        height=520, margin=dict(t=70, l=250),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.markdown("### Scatter: Select Any Variable vs Yield")

    scatter_options = [c for c in feat_cols if c in master.columns]
    selected = st.selectbox("Choose a variable to plot against yield:",
                             scatter_options,
                             index=scatter_options.index('Rain_LongRains_Fraction')
                             if 'Rain_LongRains_Fraction' in scatter_options else 0)

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        r_val = master['Yield_Tonnes_Ha'].corr(master[selected])
        z     = np.polyfit(master[selected], master['Yield_Tonnes_Ha'], 1)
        x_min, x_max = master[selected].min(), master[selected].max()
        x_line = np.linspace(x_min, x_max, 100)

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=x_line, y=np.poly1d(z)(x_line),
            mode='lines', line=dict(color=PALETTE['grey'], dash='dash', width=1.5),
            name='Trend', showlegend=False,
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
            title=f'{selected} vs Yield (r = {r_val:+.3f})',
            xaxis_title=selected,
            yaxis_title='Yield (t/ha)',
            plot_bgcolor='white', paper_bgcolor='white',
            height=380,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_s2:
        st.markdown("#### Interpretation")
        strength = "Strong" if abs(r_val) > 0.5 else ("Moderate" if abs(r_val) > 0.3 else "Weak")
        direction = "positive" if r_val > 0 else "negative"
        st.markdown(f"""
        <div class="info-box">
        <b>r = {r_val:+.4f}</b><br>
        {strength} {direction} relationship.<br><br>
        {'As this variable increases, yield tends to increase.' if r_val > 0
         else 'As this variable increases, yield tends to decrease.'}
        </div>
        """, unsafe_allow_html=True)

        if abs(r_val) < 0.3:
            st.markdown("""
            <div class="warn-box">
            Weak correlation — this variable alone is not a reliable predictor.
            The model combines multiple weak signals to improve accuracy.
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# TAB 3 — YIELD PREDICTOR
# =============================================================================

with tab3:

    st.markdown("### Interactive Yield Predictor")
    st.markdown("*Adjust the sliders in the sidebar and the prediction updates instantly.*")

    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        # Main prediction display
        pred_val  = prediction['avg']
        pred_low  = prediction['low']
        pred_high = prediction['high']

        if pred_val >= COUNTY_MEAN + 0.2:
            verdict = "Above average season"
            v_color = "#2d6a4f"
        elif pred_val <= COUNTY_MEAN - 0.2:
            verdict = "Below average season"
            v_color = "#e63946"
        else:
            verdict = "Average season"
            v_color = "#f4a261"

        st.markdown(f"""
        <div class="prediction-box">
            <p>Predicted Maize Yield</p>
            <h1>{pred_val:.2f} t/ha</h1>
            <p>Uncertainty range: {pred_low:.2f} – {pred_high:.2f} t/ha</p>
            <p style="color:white; font-size:0.85rem; margin-top:8px;">
                Random Forest: {prediction['rf']:.2f} t/ha &nbsp;|&nbsp;
                SVR: {prediction['svr']:.2f} t/ha
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_val,
            delta={'reference': COUNTY_MEAN, 'valueformat': '.2f',
                   'suffix': ' t/ha vs mean'},
            gauge={
                'axis': {'range': [2.0, 5.0], 'ticksuffix': ' t/ha'},
                'bar': {'color': PALETTE['green']},
                'steps': [
                    {'range': [2.0, 3.0], 'color': '#ffebee'},
                    {'range': [3.0, 3.5], 'color': '#fff3e0'},
                    {'range': [3.5, 4.0], 'color': '#e8f5e9'},
                    {'range': [4.0, 5.0], 'color': '#c8e6c9'},
                ],
                'threshold': {
                    'line': {'color': PALETTE['grey'], 'width': 2},
                    'thickness': 0.75,
                    'value': COUNTY_MEAN,
                },
            },
            title={'text': f"Yield Prediction<br><span style='font-size:0.8em;color:{v_color}'>"
                           f"{verdict}</span>"},
            number={'suffix': ' t/ha', 'valueformat': '.2f'},
        ))
        fig_gauge.update_layout(
            height=280,
            paper_bgcolor='white',
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_p2:
        st.markdown("#### Input Summary")

        rain_annual  = rain_lr + rain_sr + rain_off + rain_dry
        lr_fraction  = rain_lr / rain_annual if rain_annual > 0 else 0

        timing_label = "Good — rain arrived at the right time" \
                       if lr_fraction > 0.35 else \
                       "Poor — most rain fell outside planting season"
        timing_color = PALETTE['green'] if lr_fraction > 0.35 else PALETTE['red']

        # Rainfall breakdown donut
        fig_donut = go.Figure(go.Pie(
            labels=['Long Rains (Mar–May)', 'Short Rains (Sep–Nov)',
                    'Off Season (Jun–Aug)', 'Dry Season (Dec–Feb)'],
            values=[rain_lr, rain_sr, rain_off, rain_dry],
            hole=0.55,
            marker_colors=[PALETTE['green'], PALETTE['light_green'],
                           PALETTE['orange'], PALETTE['blue']],
            textinfo='label+percent',
            textfont_size=10,
            hovertemplate='<b>%{label}</b><br>%{value:.0f} mm (%{percent})<extra></extra>',
        ))
        fig_donut.update_layout(
            title=f'Rainfall Distribution — {rain_annual:.0f} mm total',
            height=280,
            paper_bgcolor='white',
            margin=dict(t=50, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown(f"""
        <div class="info-box">
        <b>Planting-season rain share:</b> {lr_fraction:.1%}<br>
        <span style="color:{timing_color}">{timing_label}</span>
        </div>
        <div class="info-box">
        <b>Total annual rainfall:</b> {rain_annual:.0f} mm<br>
        County historical range: 1,142 – 2,371 mm
        </div>
        <div class="warn-box">
        <b>Prediction uncertainty:</b> ±0.36 t/ha based on Leave-One-Out
        cross-validation across 12 years of historical data.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Compare Scenarios")

    st.markdown("Adjust the sliders above for each scenario and record results here.")

    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = []

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("Save Current Scenario"):
            st.session_state.scenarios.append({
                'Label':        f"Scenario {len(st.session_state.scenarios)+1}",
                'Long Rains mm':rain_lr,
                'Annual mm':    rain_annual,
                'LR Fraction':  f"{lr_fraction:.1%}",
                'Prediction':   pred_val,
                'Range':        f"{pred_low:.2f} – {pred_high:.2f}",
            })
    with col_btn2:
        if st.button("Clear Scenarios"):
            st.session_state.scenarios = []

    if st.session_state.scenarios:
        sc_df = pd.DataFrame(st.session_state.scenarios)
        fig_sc_bar = px.bar(
            sc_df, x='Label', y='Prediction',
            color='Prediction',
            color_continuous_scale=['#e63946', '#f4a261', '#2d6a4f'],
            range_color=[2.5, 4.5],
            text='Prediction',
        )
        fig_sc_bar.add_hline(y=COUNTY_MEAN, line_dash='dash',
                             line_color=PALETTE['grey'],
                             annotation_text=f"County mean: {COUNTY_MEAN:.2f}")
        fig_sc_bar.update_traces(texttemplate='%{text:.2f} t/ha', textposition='outside')
        fig_sc_bar.update_layout(
            title='Scenario Comparison',
            yaxis_title='Predicted Yield (t/ha)',
            plot_bgcolor='white', paper_bgcolor='white',
            coloraxis_showscale=False, height=320,
        )
        st.plotly_chart(fig_sc_bar, use_container_width=True)
        st.dataframe(sc_df, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 4 — MODEL PERFORMANCE
# =============================================================================

with tab4:

    st.markdown("### Model Validation Results")
    st.markdown("""
    <div class="info-box">
    All predictions below were made using <b>Leave-One-Out Cross-Validation</b>.
    Each year was held out, the model trained on the remaining 11 years,
    then predicted the held-out year. This is the only statistically honest
    performance estimate for a dataset of 12 observations.
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("LOO RMSE",      "0.364 t/ha",  "Random Forest")
    col_m2.metric("LOO R²",        "0.304",        "30% variance explained")
    col_m3.metric("Error % Mean",  "10.0%",        "Of 3.65 t/ha average")
    col_m4.metric("Model Bias",    "−0.007 t/ha",  "Effectively unbiased")

    st.markdown("---")

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        # Actual vs predicted scatter
        if 'Rain_LongRains_Fraction' in master.columns:
            X_all    = master[MODEL_FEATURES].copy()
            X_scaled = scaler.transform(X_all)
            rf_train = rf_model.predict(X_scaled)

        # Use pre-computed LOO error values from the analysis
        loo_preds = np.array([3.550, 3.813, 3.756, 3.750, 3.891,
                               3.422, 3.530, 3.547, 3.451, 3.804,
                               3.495, 3.690])
        actual    = master['Yield_Tonnes_Ha'].values
        years     = master['Year'].values

        fig_avp = go.Figure()
        lo = min(actual.min(), loo_preds.min()) - 0.25
        hi = max(actual.max(), loo_preds.max()) + 0.25
        fig_avp.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode='lines', line=dict(color=PALETTE['grey'], dash='dash', width=1.5),
            name='Perfect prediction', showlegend=True,
        ))
        # Error band
        fig_avp.add_trace(go.Scatter(
            x=[lo, hi], y=[lo+LOO_RMSE, hi+LOO_RMSE],
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip',
        ))
        fig_avp.add_trace(go.Scatter(
            x=[lo, hi], y=[lo-LOO_RMSE, hi-LOO_RMSE],
            mode='lines', fill='tonexty',
            fillcolor='rgba(45,106,79,0.08)',
            line=dict(width=0),
            name=f'±{LOO_RMSE:.2f} t/ha error band',
        ))
        fig_avp.add_trace(go.Scatter(
            x=actual, y=loo_preds,
            mode='markers+text',
            marker=dict(size=12, color=PALETTE['green'],
                        line=dict(color='white', width=2)),
            text=[str(y) for y in years],
            textposition='top right',
            textfont=dict(size=9),
            name='LOO prediction',
            hovertemplate='<b>%{text}</b><br>Actual: %{x:.3f} t/ha<br>'
                          'Predicted: %{y:.3f} t/ha<extra></extra>',
        ))
        fig_avp.update_layout(
            title='Actual vs Predicted Yield (LOO Cross-Validation)',
            xaxis_title='Actual Yield (t/ha)',
            yaxis_title='Predicted Yield (t/ha)',
            plot_bgcolor='white', paper_bgcolor='white',
            height=380,
            xaxis=dict(range=[lo, hi]),
            yaxis=dict(range=[lo, hi]),
        )
        st.plotly_chart(fig_avp, use_container_width=True)

    with col_v2:
        # Residuals bar chart
        residuals = loo_preds - actual
        res_colors = [PALETTE['green'] if r >= 0 else PALETTE['red'] for r in residuals]

        fig_res = go.Figure(go.Bar(
            x=years, y=residuals,
            marker_color=res_colors,
            marker_line_color='white',
            text=[f'{r:+.2f}' for r in residuals],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Error: %{y:+.3f} t/ha<extra></extra>',
        ))
        fig_res.add_hline(y=0,          line_color='black', line_width=1)
        fig_res.add_hline(y=LOO_RMSE,   line_color=PALETTE['grey'],
                          line_dash='dash', line_width=1.2,
                          annotation_text=f'+{LOO_RMSE:.2f}')
        fig_res.add_hline(y=-LOO_RMSE,  line_color=PALETTE['grey'],
                          line_dash='dash', line_width=1.2,
                          annotation_text=f'−{LOO_RMSE:.2f}')
        fig_res.update_layout(
            title='Prediction Errors by Year<br>'
                  '<sup>Green = over-predicted &nbsp;|&nbsp; Red = under-predicted</sup>',
            xaxis_title='Year', yaxis_title='Error (t/ha)',
            xaxis=dict(tickmode='array', tickvals=years.tolist()),
            plot_bgcolor='white', paper_bgcolor='white',
            height=380,
        )
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")

    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        # Feature importance
        perm_importance = {
            'Humidity_ShortRains_pct': 0.0290,
            'Rain_DrySeason_mm':       0.0287,
            'Rain_LongRains_Fraction': 0.0249,
            'Temp_ShortRains_C':       0.0132,
            'Rain_OffSeason_mm':       0.0073,
            'SoilWetness_ShortRains':  0.0041,
        }
        imp_df = pd.DataFrame(list(perm_importance.items()),
                              columns=['Feature', 'Importance'])
        imp_df = imp_df.sort_values('Importance')

        fig_imp = go.Figure(go.Bar(
            x=imp_df['Importance'],
            y=imp_df['Feature'],
            orientation='h',
            marker_color=PALETTE['blue'],
            marker_line_color='white',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
        ))
        fig_imp.update_layout(
            title='Variable Importance (Permutation Method)<br>'
                  '<sup>Longer bar = removing this variable hurts accuracy more</sup>',
            xaxis_title='Importance Score',
            plot_bgcolor='white', paper_bgcolor='white',
            height=320,
            margin=dict(l=200),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_f2:
        st.markdown("### Reliability Assessment")
        reliability = {
            'Data engineering approach':  ('★★★★★', 'Seasonal design is sound'),
            'Leakage prevention':         ('★★★★★', 'No target leakage'),
            'Validation method':          ('★★★★★', 'LOO correct for n=12'),
            'Variable selection':         ('★★★★☆', '6 variables well justified'),
            'Model choice':               ('★★★★☆', 'RF + SVR appropriate'),
            'Confidence in results':      ('★★★☆☆', 'Directional patterns reliable'),
            'Data quantity':              ('★★☆☆☆', 'n=12 limits everything'),
        }
        rel_df = pd.DataFrame(
            [(k, v[0], v[1]) for k, v in reliability.items()],
            columns=['Aspect', 'Rating', 'Notes']
        )
        st.dataframe(rel_df, use_container_width=True, hide_index=True, height=280)

        st.markdown("""
        <div class="warn-box">
        <b>Overall assessment:</b> Valid for identifying key yield drivers
        and scenario planning. Not suitable for operational forecasting
        without additional historical data.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Full Prediction Table (LOO)")
    pred_table = pd.DataFrame({
        'Year':          years,
        'Actual (t/ha)': actual.round(3),
        'Predicted':     loo_preds.round(3),
        'Error (t/ha)':  residuals.round(3),
        'Variety':       master['Variety'].values,
    })
    st.dataframe(
        pred_table.style.format({
            'Actual (t/ha)': '{:.3f}',
            'Predicted':     '{:.3f}',
            'Error (t/ha)':  '{:+.3f}',
        }).background_gradient(subset=['Actual (t/ha)'], cmap='Greens'),
        use_container_width=True, hide_index=True,
    )

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding:8px 0;">
    Maize Yield Prediction — Uasin Gishu County, Kenya &nbsp;|&nbsp;
    IBM SkillsBuild Data Analytics Bootcamp &nbsp;|&nbsp;
    Data: Ministry of Agriculture, NASA POWER, CIMMYT, Purdue University
</div>
""", unsafe_allow_html=True)
