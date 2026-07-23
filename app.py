import streamlit as st
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
            win_prob = predict_game(opponent_abbr, is_home=(location == "Home"), players_out=manual_out)
        except FileNotFoundError:
            st.error("No trained model found. Run `python3 predict_upcoming.py` once first "
                     "to train and save thunder_model.joblib.")
            st.stop()
        except Exception as e:
            st.error(f"Couldn't reach NBA.com's live stats API ({e}). This can happen on "
                     f"cloud-hosted servers that NBA.com occasionally blocks -- if this "
                     f"persists on the deployed app, it likely still works when run locally.")
            st.stop()

    st.metric("OKC Win Probability", f"{win_prob:.1%}")
    st.progress(win_prob)

    if win_prob >= 0.6:
        st.success(f"Model favors OKC to beat the {opponent_name}.")
    elif win_prob <= 0.4:
        st.warning(f"Model favors the {opponent_name} in this matchup.")
    else:
        st.info("Model sees this as close to a coin flip.")

st.divider()
st.caption("See the **Model Validation** page (sidebar) for backtest accuracy, feature "
           "importance, and methodology notes.")
