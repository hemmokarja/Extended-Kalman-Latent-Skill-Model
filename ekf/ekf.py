import itertools
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


@dataclass
class EKFConfig:
    sigma_match: float = 1.0  # match outcome noise parameter
    lambda_decay: float = 0.02  # momentum decay rate (lambda > 0), half life ln(2)/0.02 ≈ 30 days
    sigma_skill: float = 0.05  # process noise std dev for skill
    sigma_momentum: float = 0.05  # process noise std dev for momentum
    default_skill: float = 0.0  # initial skill for new players
    default_skill_var: float = 3.0  # initial skill uncertainty
    default_momentum: float = 0.0  # initial momentum
    default_momentum_var: float = 2.0  # initial momentum uncertainty
    default_covariance: float = 0.0  # initial skill-momentum covariance


@dataclass
class Match:
    player1: str
    player2: str
    winner: str
    player1_time_delta: float
    player2_time_delta: float
    match_index: int

    def __post_init__(self):
        if self.winner not in [self.player1, self.player2]:
            raise ValueError(f"Winner must be either {self.player1} or {self.player2}")


@dataclass
class PredictedMatch(Match):
    pred_proba: float

    @classmethod
    def from_match(cls, match: Match, pred_proba: float) -> "PredictedMatch":
        return cls(
            player1=match.player1,
            player2=match.player2,
            winner=match.winner,
            player1_time_delta=match.player1_time_delta,
            player2_time_delta=match.player2_time_delta,
            match_index=match.match_index,
            pred_proba=pred_proba,
        )


class Matches:
    def __init__(self, matches: List[Union[Match, PredictedMatch]]):
        self.matches = matches

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "Matches":
        required_cols = [
            "player1",
            "player2", 
            "winner",
            "player1_time_delta",
            "player2_time_delta",
            "match_index",
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")

        matches = []
        for _, row in df.iterrows():
            match = Match(
                player1=str(row.player1),
                player2=str(row.player2),
                winner=str(row.winner),
                player1_time_delta=float(row.player1_time_delta),
                player2_time_delta=float(row.player2_time_delta),
                match_index=int(row.match_index),
            )
            matches.append(match)

        return cls(matches)

    def to_pandas(self) -> pd.DataFrame:
        records = []
        for match in self.matches:
            record = {
                "player1": match.player1,
                "player2": match.player2,
                "winner": match.winner,
                "player1_time_delta": match.player1_time_delta,
                "player2_time_delta": match.player2_time_delta,
                "match_index": match.match_index,
            }
            if isinstance(match, PredictedMatch):
                record["pred_proba"] = match.pred_proba
            records.append(record)
        return pd.DataFrame(records)

    def __len__(self) -> int:
        return len(self.matches)
    
    def __getitem__(self, index: int) -> Match:
        return self.matches[index]
    
    def __iter__(self):
        return iter(self.matches)


@dataclass
class PlayerState:
    skill_mean: float
    skill_variance: float
    momentum_mean: float
    momentum_variance: float
    skill_momentum_covariance: float

    @property
    def effective_skill(self) -> float:
        # total skill = skill mean + momentum
        return self.skill_mean + self.momentum_mean

    @property
    def effective_skill_variance(self) -> float:
        return (
            self.skill_variance
            + self.momentum_variance
            + 2 * self.skill_momentum_covariance
        )


class PlayerHistory:
    def __init__(self, player_id: str, initial_state: PlayerState):
        self.player_id = player_id
        self.states: Dict[int, PlayerState] = {-1: initial_state}  # -1 initial state
        self.match_indices = [-1]  # track which indices we have states for

    def add_state(self, match_index: int, state: PlayerState):
        if match_index < max(self.match_indices):
            raise ValueError(
                f"match_index {match_index} must be greater than all previous "
                f"indices {self.match_indices}"
            )
        self.states[match_index] = state
        self.match_indices.append(match_index)

    def get_prematch_state(self, match_index: int) -> PlayerState:
        if match_index < 0:
            raise ValueError(f"match_index must greater than 0, got {match_index}")

        valid_indices = [idx for idx in self.match_indices if idx < match_index]
        most_recent_index = max(valid_indices)
        return self.states[most_recent_index]

    def current_state(self) -> PlayerState:
        """Get the most recent state."""
        latest_index = max(self.match_indices)
        return self.states[latest_index]

    def get_all_prematch_states_and_indices(self) -> List[Tuple[int, PlayerState]]:
        prematch_states = []
        match_indices = []
        for match_index in self.match_indices:
            if match_index == -1:
                continue
            prematch_states.append(self.get_prematch_state(match_index))
            match_indices.append(match_index)
        return prematch_states, match_indices


class SkillHistory:
    def __init__(self):
        self.players: Dict[str, PlayerHistory] = {}
        self.match_count = 0

    def add_player(self, player_id: str, initial_state: PlayerState):
        self.players[player_id] = PlayerHistory(player_id, initial_state)

    def update_player(self, player_id: str, match_index: int, new_state: PlayerState):
        if player_id not in self.players:
            raise ValueError(f"Player {player_id} not found in history")
        self.players[player_id].add_state(match_index, new_state)
        self.match_count = max(self.match_count, match_index + 1)

    def get_player_state(
        self, player_id: str, match_index: Optional[int] = None
    ) -> PlayerState:
        if player_id not in self.players:
            raise ValueError(f"Player {player_id} not found in history")

        if match_index is None:
            print("No match index provided, using latest state")
            return self.players[player_id].current_state()
        else:
            return self.players[player_id].get_prematch_state(match_index)

    def get_all_current_states(self) -> Dict[str, PlayerState]:
        return {pid: history.current_state() for pid, history in self.players.items()}

    def plot_history(
        self,
        player_ids: Optional[List[str]] = None,
        show: str = "effective",  # "effective" or "decomposed"
        confidence: Optional[float] = None,  # e.g., 0.95 for 95% CI
        show_all: bool = True,  # whether to show other players in gray background
    ):
        if isinstance(player_ids, str):
            player_ids = [player_ids]

        # determine which players to highlight vs show in background
        all_player_ids = list(self.players.keys())
        if player_ids is None:
            highlighted_players = all_player_ids
            background_players = []
        else:
            highlighted_players = player_ids
            background_players = [
                pid for pid in all_player_ids if pid not in player_ids
            ] if show_all else []

        # get z-score for confidence interval
        z_score = st.norm.ppf(0.5 + confidence / 2) if confidence else None
        color_cycle = itertools.cycle(plt.colormaps["tab10"].colors)

        if show == "effective":
            plt.figure(figsize=(12, 6))
            
            # Plot background players in gray
            for player_id in background_players:
                if player_id not in self.players:
                    continue
                states, match_indices = (
                    self.players[player_id].get_all_prematch_states_and_indices()
                )
                means = [s.effective_skill for s in states]
                plt.plot(match_indices, means, color="gray", alpha=0.15, linewidth=1)
            
            # Plot highlighted players with colors
            for player_id in highlighted_players:
                if player_id not in self.players:
                    print(f"Skipping missing player: {player_id}")
                    continue
                
                states, match_indices = (
                    self.players[player_id].get_all_prematch_states_and_indices()
                )
                color = next(color_cycle)
                
                means = [s.effective_skill for s in states]
                plt.plot(
                    match_indices, means, label=f"{player_id}", color=color, linewidth=1
                )
                
                if confidence:
                    stds = [np.sqrt(s.effective_skill_variance) for s in states]
                    margin = np.array(stds) * z_score
                    lower = np.array(means) - margin
                    upper = np.array(means) + margin
                    plt.fill_between(
                        match_indices, lower, upper, color=color, alpha=0.1
                    )
            
            plt.xlabel("Match index")
            plt.ylabel("Skill")
            plt.title("Player Skill Trajectories")
            plt.legend(loc="best")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
        elif show == "decomposed":
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # plot background players in gray on both subplots
            for player_id in background_players:
                if player_id not in self.players:
                    continue
                states, match_indices = (
                    self.players[player_id].get_all_prematch_states_and_indices()
                )
                
                skill_means = [s.skill_mean for s in states]
                momentum_means = [s.momentum_mean for s in states]

                ax1.plot(
                    match_indices, skill_means, color="gray", alpha=0.3, linewidth=1
                )
                ax2.plot(
                    match_indices, momentum_means, color="gray", alpha=0.3, linewidth=1
                )

            # plot highlighted players with colors
            for player_id in highlighted_players:
                if player_id not in self.players:
                    print(f"Skipping missing player: {player_id}")
                    continue

                states, match_indices = (
                    self.players[player_id].get_all_prematch_states_and_indices()
                )
                color = next(color_cycle)
                
                # skill subplot
                skill_means = [s.skill_mean for s in states]
                ax1.plot(
                    match_indices,
                    skill_means,
                    label=f"{player_id}",
                    color=color,
                    linewidth=1,
                )
                
                if confidence:
                    skill_stds = [np.sqrt(s.skill_variance) for s in states]
                    margin = np.array(skill_stds) * z_score
                    lower = np.array(skill_means) - margin
                    upper = np.array(skill_means) + margin
                    ax1.fill_between(
                        match_indices, lower, upper, color=color, alpha=0.1
                    )

                # momentum subplot
                momentum_means = [s.momentum_mean for s in states]
                ax2.plot(
                    match_indices,
                    momentum_means,
                    label=f"{player_id}",
                    color=color,
                    linewidth=1,
                )

                if confidence:
                    momentum_stds = [np.sqrt(s.momentum_variance) for s in states]
                    margin = np.array(momentum_stds) * z_score
                    lower = np.array(momentum_means) - margin
                    upper = np.array(momentum_means) + margin
                    ax2.fill_between(
                        match_indices, lower, upper, color=color, alpha=0.1
                    )

            ax1.set_ylabel("Skill")
            ax1.set_title("Player Skill Trajectories")
            ax1.legend(loc="best")
            ax1.grid(True)
            
            ax2.set_xlabel("Match index")
            ax2.set_ylabel("Momentum")
            ax2.set_title("Player Momentum Trajectories")
            ax2.legend(loc="best")
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        else:
            raise ValueError("Invalid show argument: use 'effective' or 'decomposed'")


class EKFSkillRating:
    def __init__(self, config: EKFConfig):
        self.config = config
        self.history = None
        self.logger = logging.getLogger(__name__)

    def _create_initial_state(self) -> PlayerState:
        return PlayerState(
            skill_mean=self.config.default_skill,
            skill_variance=self.config.default_skill_var,
            momentum_mean=self.config.default_momentum,
            momentum_variance=self.config.default_momentum_var,
            skill_momentum_covariance=self.config.default_covariance
        )

    def _time_propagation(self, state: PlayerState, dt: float) -> PlayerState:

        # state transition matrix
        F = np.array([
            [1.0, 0.0],  # skill update
            [0.0, np.exp(-self.config.lambda_decay * dt)]  # momentum decay
        ])
        
        # current state vector
        x = np.array([state.skill_mean, state.momentum_mean])
        
        # current covariance matrix
        P = np.array([
            [state.skill_variance, state.skill_momentum_covariance],
            [state.skill_momentum_covariance, state.momentum_variance]
        ])
        
        # process noise
        Q = np.array([
            [self.config.sigma_skill**2 * dt, 0.0],
            [0.0, self.config.sigma_momentum**2 * dt]
        ])
        
        # propagate state and covariance
        x_new = F @ x
        P_new = F @ P @ F.T + Q

        return PlayerState(
            skill_mean=float(x_new[0]),
            skill_variance=float(P_new[0, 0]),
            momentum_mean=float(x_new[1]),
            momentum_variance=float(P_new[1, 1]),
            skill_momentum_covariance=float(P_new[0, 1])
        )

    def _ekf_update(
        self, state1: PlayerState, state2: PlayerState, player1_wins: bool
    ) -> Tuple[PlayerState, PlayerState]:

        # calculate prediction
        delta = (
            (state1.effective_skill - state2.effective_skill) / self.config.sigma_match
        )
        pred_prob = st.norm.cdf(delta)
        pdf = st.norm.pdf(delta)

        # state vectors
        x = np.array([state1.skill_mean, state1.momentum_mean, 
                     state2.skill_mean, state2.momentum_mean])

        # prior covariance matrix
        P = np.array([
            [state1.skill_variance, state1.skill_momentum_covariance, 0.0, 0.0],
            [state1.skill_momentum_covariance, state1.momentum_variance, 0.0, 0.0],
            [0.0, 0.0, state2.skill_variance, state2.skill_momentum_covariance],
            [0.0, 0.0, state2.skill_momentum_covariance, state2.momentum_variance]
        ])

        # jacobian matrix
        H = np.array([[pdf/self.config.sigma_match, pdf/self.config.sigma_match,
                      -pdf/self.config.sigma_match, -pdf/self.config.sigma_match]])

        # observation noise
        R = pred_prob * (1 - pred_prob)

        # kalman gain computation
        eps = 1e-6
        S = H @ P @ H.T + R + eps
        K = P @ H.T / S

        # innovation
        y = 1.0 if player1_wins else 0.0
        innovation = y - pred_prob

        # state update
        x_new = x + (K.flatten() * innovation)

        # covariance update
        P_new = P - K @ H @ P

        new_state1 = PlayerState(
            skill_mean=float(x_new[0]), 
            skill_variance=float(P_new[0, 0]),
            momentum_mean=float(x_new[1]),
            momentum_variance=float(P_new[1, 1]),
            skill_momentum_covariance=float(P_new[0, 1])
        )

        new_state2 = PlayerState(
            skill_mean=float(x_new[2]),
            skill_variance=float(P_new[2, 2]),
            momentum_mean=float(x_new[3]),
            momentum_variance=float(P_new[3, 3]),
            skill_momentum_covariance=float(P_new[2, 3])
        )

        return new_state1, new_state2

    def fit(self, matches: Matches) -> SkillHistory:

        start = time.time()

        self.history = SkillHistory()

        for match in matches:

            # ensure players exist in history
            for player in [match.player1, match.player2]:
                if player not in self.history.players:
                    self.history.add_player(player, self._create_initial_state())

            # get most recent states before this match
            state1 = self.history.get_player_state(match.player1, match.match_index)
            state2 = self.history.get_player_state(match.player2, match.match_index)

            # time propagation
            state1 = self._time_propagation(state1, match.player1_time_delta)
            state2 = self._time_propagation(state2, match.player2_time_delta)

            # EKF update
            player1_wins = match.winner == match.player1
            new_state1, new_state2 = self._ekf_update(state1, state2, player1_wins)

            # update history with new states
            self.history.update_player(match.player1, match.match_index, new_state1)
            self.history.update_player(match.player2, match.match_index, new_state2)
        
        end = time.time()
        self.logger.info(
            f"Fitting {len(matches)} matches completed in {end - start:.2f} seconds"
        )

        return self.history

    def predict_proba(
        self,
        player1: str,
        player2: str, 
        player1_time_delta: float = 0.0,
        player2_time_delta: float = 0.0,
        match_index: Optional[int] = None,
    ) -> float:
        """Predict probability that player1 beats player2."""
        if self.history is None:
            raise ValueError("Model must be fitted before making predictions")

        state1 = self.history.get_player_state(player1, match_index)
        state2 = self.history.get_player_state(player2, match_index)

        state1 = self._time_propagation(state1, player1_time_delta)
        state2 = self._time_propagation(state2, player2_time_delta)

        delta = (
            (state1.effective_skill - state2.effective_skill) / self.config.sigma_match
        )
        return st.norm.cdf(delta)

    def predict_matches(self, matches: Matches) -> Matches:
        if self.history is None:
            raise ValueError("Model must be fitted before making predictions")

        predicted = []
        for match in matches:
            prob = self.predict_proba(
                player1=match.player1,
                player2=match.player2,
                player1_time_delta=match.player1_time_delta,
                player2_time_delta=match.player2_time_delta,
                match_index=match.match_index,
            )
            predicted_match = PredictedMatch.from_match(match, prob)
            predicted.append(predicted_match)

        return Matches(predicted)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = [
            "player1",
            "player2",
            "player1_time_delta",
            "player2_time_delta",
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")

        df_copy = df.copy()

        def _predict_row(row):
            return self.predict_proba(
                player1=str(row["player1"]),
                player2=str(row["player2"]),
                player1_time_delta=float(row["player1_time_delta"]),
                player2_time_delta=float(row["player2_time_delta"]),
                match_index=row.get("match_index", None),
            )

        df_copy["pred_proba"] = df_copy.apply(_predict_row, axis=1)
        return df_copy

    def get_current_ratings(self) -> Dict[str, PlayerState]:
        if self.history is None:
            raise ValueError("Model must be fitted before getting ratings")
        return self.history.get_all_current_states()
