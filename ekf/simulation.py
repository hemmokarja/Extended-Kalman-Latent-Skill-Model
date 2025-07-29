import datetime

import numpy as np
import pandas as pd
import scipy.stats as st


def simulate_matches(n_players, n_matches, random_state=None):
    """
    Simulates a sequence of 1-vs-1 matches between players, generating synthetic data
    for use with an EKF latent skill model.

    Each player has a latent "true skill" that evolves slightly over time based on
    time since their last match. Match outcomes are binary (win/loss) and are 
    probabilistically determined based on the difference in player skills and 
    Gaussian match noise.
    """
    np.random.seed(random_state)

    player_names = [f"Player-{i}" for i in range(n_players)]
    player_last_match = {name: None for name in player_names}

    true_skills = np.random.normal(0, 1, n_players)
    match_date = datetime.date(2020, 1, 1)

    records = []

    for i in range(n_matches):

        p1_idx, p2_idx = np.random.choice(n_players, 2, replace=False)
        p1, p2 = player_names[p1_idx], player_names[p2_idx]

        # current match date (roughly 2-3 days between matches on average)
        match_date = match_date + datetime.timedelta(days=np.random.randint(0, 4))

        p1_time_delta = (
            30 if player_last_match[p1] is None
            else (match_date - player_last_match[p1]).days
        )
        p2_time_delta = (
            30 if player_last_match[p2] is None
            else (match_date - player_last_match[p2]).days
        )

        # evolve true skills slightly with time (for data generation)
        true_skills[p1_idx] += np.random.normal(0, 0.05 * np.sqrt(p1_time_delta / 30))
        true_skills[p2_idx] += np.random.normal(0, 0.05 * np.sqrt(p2_time_delta / 30))

        # simulate match outcome
        skill_diff = true_skills[p1_idx] - true_skills[p2_idx]
        win_prob = st.norm.cdf(skill_diff / 0.8)  # 0.8 is match noise
        winner = p1 if np.random.random() < win_prob else p2

        records.append(
            {
                "match_index": i,
                "date": match_date,
                "player1": p1,
                "player2": p2,
                "winner": winner,
                "player1_time_delta": p1_time_delta,
                "player2_time_delta": p2_time_delta,
                "player1_true_skill": true_skills[p1_idx],
                "player2_true_skill": true_skills[p2_idx],
            }
        )

        player_last_match[p1] = match_date
        player_last_match[p2] = match_date

    matches_df = pd.DataFrame(records)
    return matches_df
