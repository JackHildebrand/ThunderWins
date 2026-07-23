import streamlit as st
import plotly.graph_objects as go
from nba_api.stats.static import teams as nba_teams

from predict_upcoming import predict_game
from dataset import PRODUCTION_PLAYERS

st.set_page_config(page_title="OKC Thunder Game Predictor", page_icon="⚡", layout="centered")

st.title("⚡ OKC Thunder Game Predictor")
st.caption("A Random Forest model predicting Thunder win probability for 2026-27 games, "
           "trained on 7 seasons of team, opponent, and player-level data.")

all_teams = sorted(
    [t for t in nba_teams.get_teams() if t['abbreviation'] != 'OKC'],
    key=lambda t: t['full_name']
)
team_options = {t['full_name']: t['abbreviation'] for t in all_teams}

col1, col2 = st.columns(2)
with col1:
    opponent_name = st.selectbox("Opponent", list(team_options.keys()),
                                  index=list(team_options.keys()).index("Denver Nuggets"))
with col2:
    location = st.radio("Location", ["Home", "Away"], horizontal=True)

st.subheader("Injuries")
st.caption("Checkboxes here force a player OUT for this prediction. The model also "
           "auto-checks ESPN's live injury feed on its own -- these are just a manual override.")

manual_out = []
cols = st.columns(len(PRODUCTION_PLAYERS))
for i, (full_name, label) in enumerate(PRODUCTION_PLAYERS):
    with cols[i]:
        if st.checkbox(full_name.split()[-1], key=f"out_{label}"):
            manual_out.append(label)

if st.button("Predict", type="primary", use_container_width=True):
    opponent_abbr = team_options[opponent_name]
    with st.spinner("Pulling live data and predicting..."):
        try:
            result = predict_game(opponent_abbr, is_home=(location == "Home"), players_out=manual_out)
        except FileNotFoundError:
            st.error("No trained model found. Run `python3 predict_upcoming.py` once first "
                     "to train and save thunder_model.joblib.")
            st.stop()
        except Exception as e:
            st.error(f"Couldn't reach NBA.com's live stats API ({e}). This can happen on "
                     f"cloud-hosted servers that NBA.com occasionally blocks -- if this "
                     f"persists on the deployed app, it likely still works when run locally.")
            st.stop()

    win_prob = result['win_prob']
    margin = result['predicted_margin']

    m1, m2 = st.columns(2)
    m1.metric("OKC Win Probability", f"{win_prob:.1%}")
    m2.metric("Predicted Margin", f"{'OKC +' if margin >= 0 else 'OKC '}{margin:.1f}")
    st.progress(win_prob)

    if win_prob >= 0.6:
        st.success(f"Model favors OKC to beat the {opponent_name}.")
    elif win_prob <= 0.4:
        st.warning(f"Model favors the {opponent_name} in this matchup.")
    else:
        st.info("Model sees this as close to a coin flip.")

    st.caption("Win probability and margin come from two separately trained models "
               "(a classifier and a regressor), so they can occasionally point in "
               "slightly different directions on close games -- that's expected, not a bug.")

    st.subheader("Why the model predicted this")
    st.caption("SHAP values for this specific prediction -- how much each factor pushed "
               "the win probability up (blue) or down (red) from the model's average "
               "prediction. This is different from the Model Validation page's feature "
               "importance, which shows what matters *in general*, not for this one game.")

    top_contributions = result['contributions'].head(8).sort_values()
    colors = ['#d62728' if v < 0 else '#1f77b4' for v in top_contributions.values]
    fig = go.Figure(go.Bar(
        x=top_contributions.values,
        y=top_contributions.index,
        orientation='h',
        marker_color=colors,
    ))
    fig.update_layout(
        xaxis_title="Impact on win probability",
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("See the **Model Validation** page (sidebar) for backtest accuracy, feature "
           "importance, and methodology notes.")
