import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def _github_to_raw(url: str) -> str:
    """
    Convert a GitHub file URL to its raw URL.

    Example:
    https://github.com/user/repo/blob/branch/path/to/file
    -> https://raw.githubusercontent.com/user/repo/branch/path/to/file
    """
    if not url.startswith("https://github.com/"):
        raise ValueError("Not a valid GitHub URL")

    parts = url[len("https://github.com/"):].split("/")

    if len(parts) < 5 or parts[2] != "blob":
        raise ValueError("URL format is not a valid GitHub file URL")

    user, repo, _, branch = parts[:4]
    file_path = "/".join(parts[4:])

    raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{file_path}"
    return raw_url


def _read_csv_from_github(url: str) -> pd.DataFrame:
    raw_url = _github_to_raw(url)
    return pd.read_csv(raw_url, encoding="latin1")


def _read_all_historical_matches_from_tml_db(
    from_year: int, to_year: int
) -> pd.DataFrame:
    """
    Read historical matches from TML Database at url:
    https://github.com/Tennismylife/TML-Database
    """
    matches = []
    for year in tqdm(range(from_year, to_year + 1), desc="Downloading matches..."):
        url = f"https://github.com/Tennismylife/TML-Database/blob/master/{year}.csv"
        try:
            year_matches = _read_csv_from_github(url)
        except Exception as e:
            print(f"Error loading data from {url}: {e}")
            raise
        matches.append(year_matches)
    return pd.concat(matches)


def _get_time_delta(player, match_date, last_match_dates, max_delta_days=365):
    last_date = last_match_dates[player]
    if last_date is None:
        return max_delta_days
    return (match_date - last_date).days


def _parse_date(date_val):
    # Converts a numeric date like 19900101 (int/float) to a Python datetime.date
    if date_val is None:
        return None
    try:
        date_str = str(int(date_val))
        return datetime.datetime.strptime(date_str, "%Y%m%d").date()
    except (ValueError, TypeError):
        return None


def _clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()

    matches["date"] = matches.tourney_date.apply(_parse_date)
    matches = matches[matches.date.notna()]

    matches["winner_is_p1"] = np.random.rand(matches.shape[0]) < 0.5
    wp1 = matches.winner_is_p1

    matches["player1"] = np.where(wp1, matches.winner_name, matches.loser_name)
    matches["player1_rank"] = np.where(wp1, matches.winner_rank, matches.loser_rank)

    matches["player2"] = np.where(~wp1, matches.winner_name, matches.loser_name)
    matches["player2_rank"] = np.where(~wp1, matches.winner_rank, matches.loser_rank)

    matches["winner"] = matches.winner_name

    # add time deltas since last match for each player
    last_match_dates = defaultdict(lambda: None)
    p1_deltas = []
    p2_deltas = []
    for _, row in matches.iterrows():
        p1_deltas.append(_get_time_delta(row.player1, row.date, last_match_dates))
        p2_deltas.append(_get_time_delta(row.player2, row.date, last_match_dates))

        last_match_dates[row.player1] = row.date
        last_match_dates[row.player2] = row.date

    matches["player1_time_delta"] = p1_deltas
    matches["player2_time_delta"] = p2_deltas

    matches["match_index"] = range(matches.shape[0])

    cols = [
        "date",
        "match_index",
        "player1",
        "player2", 
        "winner",
        "player1_time_delta",
        "player2_time_delta",
        "player1_rank",
        "player2_rank",
    ]
    return matches[cols]


def load_historical_matches(from_year: int, to_year:int ) -> pd.DataFrame:
    matches_df = _read_all_historical_matches_from_tml_db(from_year, to_year)
    matches_df = _clean_matches(matches_df)
    print(
        f"Loaded {matches_df.shape[0]} matches from {from_year} to {to_year}"
    )
    return matches_df
