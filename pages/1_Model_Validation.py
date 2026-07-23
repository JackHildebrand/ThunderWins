import streamlit as st
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Model Validation", page_icon="📊", layout="centered")
st.title("📊 Model Validation & Methodology")

st.markdown("""
This model predicts OKC Thunder game outcomes using a **Random Forest classifier**
trained on 7 seasons (2019-20 through 2025-26) of team, opponent, and player-level
data pulled live from NBA.com's stats API.

**Key methodology choices:**
- **Time-based train/test splits** -- always trained on earlier seasons, tested on a
  later one. Random shuffling would leak future information into training, which is
  a common beginner mistake with time-series sports data.
- **Shifted rolling averages** -- every "recent form" feature only uses games
  *before* the one being predicted, so a game's own result never leaks into its own
  prediction.
- **Recency weighting** -- older seasons count less during training, since the
  roster changes meaningfully year to year.
- **Correlation-based feature trimming** -- validated empirically, not assumed:
  team-level redundant features were dropped after trimming measurably improved
  backtest accuracy; the same approach was tried on player features and made
  results *worse*, so those were kept as-is instead.
""")

# Precomputed once (see build_validation_results.py) rather than recomputed live --
# this backtest doesn't change between visits, and recomputing it here would mean
# ~35 live NBA.com API calls on every cold start, which is slow and depends on
# NBA.com not blocking whatever server this happens to be deployed on.
try:
    saved = joblib.load('validation_results.joblib')
except FileNotFoundError:
    st.error("validation_results.joblib not found. Run `python3 build_validation_results.py` first.")
    st.stop()

rf_results, importances = saved['rf_results'], saved['importances']
spread_results = saved.get('spread_results')

st.subheader("Backtest Accuracy vs. Baseline")
st.caption("Baseline = always predicting the majority outcome (OKC's actual win rate that season). "
           "Beating it means the model found real signal, not just guessing the favorite.")

seasons = [r[0] for r in rf_results]
acc = [r[1] for r in rf_results]
baseline = [r[2] for r in rf_results]

fig = go.Figure()
fig.add_trace(go.Bar(name='Model Accuracy', x=seasons, y=acc))
fig.add_trace(go.Bar(name='Baseline', x=seasons, y=baseline))
fig.update_layout(barmode='group', yaxis_tickformat='.0%', yaxis_range=[0, 1])
st.plotly_chart(fig, use_container_width=True)

if spread_results:
    st.subheader("Point Margin (Spread) Accuracy")
    st.caption("A separate regression model predicting the actual point margin, not just "
               "win/loss. Baseline = always predicting the training set's average margin. "
               "Lower is better (mean absolute error, in points).")
    sp_seasons = [r[0] for r in spread_results]
    mae = [r[1] for r in spread_results]
    sp_baseline = [r[2] for r in spread_results]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name='Model MAE', x=sp_seasons, y=mae))
    fig2.add_trace(go.Bar(name='Baseline MAE', x=sp_seasons, y=sp_baseline))
    fig2.update_layout(barmode='group', yaxis_title='Mean Absolute Error (points)')
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Notably, the spread model beats its baseline in every season tested, "
               "including 2024-25 -- where the win/loss classifier struggles most. Predicting "
               "a continuous margin turned out to be more robust to that season's distribution "
               "shift than predicting the binary outcome.")

st.subheader("Feature Importance")
st.caption("Which inputs the model relies on most, from the final trained Random Forest.")
st.bar_chart(importances)

st.subheader("Known Limitation")
st.markdown("""
2024-25 is the weakest backtest season by a wide margin -- the Thunder jumped from a
69.6% win rate in 2023-24 to 82.1% in 2024-25, a level of dominance nothing in the
training data had shown before. That's genuine **distribution shift**: no amount of
feature engineering fully closes the gap when an event is truly out-of-distribution.
Adding OKC's own prior-season record and player-level continuity features
(tracked independently of season boundaries) measurably helped, but didn't fully
close it -- which is itself an honest, expected result, not a bug.
""")
