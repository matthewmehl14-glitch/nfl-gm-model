import os
import json
import csv
import requests
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime
from zoneinfo import ZoneInfo

# ============================================================
# NFL MATCHUP PREDICTION ENGINE
# ============================================================

PRIOR_SEASON = 2025
CURRENT_SEASON = 2026

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
SPORT = "americanfootball_nfl"
REGIONS = "us"
MARKETS = "h2h,spreads,totals"
BOOKMAKER = "pinnacle"
DISPLAY_TIMEZONE = "America/Chicago"

NUM_SIMS = 10000
RANDOM_SEED = 42
EPA_PRIOR_PLAYS = 400.0

DEFAULT_EPA_TO_POINTS_PER_PLAY = 0.70
MAX_MATCHUP_ADJUSTMENT_POINTS = 12.0
DEFAULT_SCORE_SD = 9.5
DEFAULT_SCORE_CORRELATION = 0.15

ML_MARKET_WEIGHT = 0.85
SPREAD_MARKET_WEIGHT = 0.60
TOTAL_MARKET_WEIGHT = 0.65

ODDS_API_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

ESPN_TEAM_MAPPING = {
    "Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF",
    "Panthers": "CAR", "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE",
    "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
    "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC",
    "Raiders": "LV", "Chargers": "LAC", "Rams": "LAR", "Dolphins": "MIA",
    "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
    "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "49ers": "SF",
    "Seahawks": "SEA", "Buccaneers": "TB", "Titans": "TEN", "Commanders": "WAS",
}

def safe_float(value, default=np.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def is_regular_season(df):
    if df.empty: return df
    if "game_type" in df.columns: return df[df["game_type"].eq("REG")].copy()
    return df.copy()

def american_to_decimal(odds):
    odds = float(odds)
    if odds > 0: return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)

def proportional_devig(odds1, odds2):
    p1 = 1.0 / american_to_decimal(odds1)
    p2 = 1.0 / american_to_decimal(odds2)
    total = p1 + p2
    if total <= 0: return 0.5, 0.5
    return p1 / total, p2 / total

def prob_to_american(pct):
    p = float(pct)
    if p <= 0: return "+99999"
    if p >= 100: return "-99999"
    decimal_fair = 100.0 / p
    if decimal_fair >= 2.0: return f"+{int(round((decimal_fair - 1.0) * 100.0))}"
    return f"{int(round(-100.0 / (decimal_fair - 1.0)))}"

def calculate_ev(win_prob, odds, push_prob=0.0):
    dec_odds = american_to_decimal(odds)
    win_p = max(0.0, min(1.0, float(win_prob) / 100.0))
    push_p = max(0.0, min(1.0 - win_p, float(push_prob) / 100.0))
    loss_p = max(0.0, 1.0 - win_p - push_p)
    profit_on_win = dec_odds - 1.0
    return ((win_p * profit_on_win) - loss_p) * 100.0

def weighted_current_value(current_value, prior_value, current_plays, prior_strength=EPA_PRIOR_PLAYS):
    current_value = safe_float(current_value, np.nan)
    prior_value = safe_float(prior_value, np.nan)
    if np.isnan(prior_value) and np.isnan(current_value): return 0.0
    if np.isnan(prior_value): return 0.0 if np.isnan(current_value) else float(current_value)
    if np.isnan(current_value) or current_plays <= 0: return float(prior_value)
    weight = current_plays / (current_plays + prior_strength)
    weight = max(0.0, min(1.0, weight))
    return float((weight * current_value) + ((1.0 - weight) * prior_value))

def get_final_game_scores(pbp):
    required = {"game_id", "home_team", "away_team", "total_home_score", "total_away_score"}
    if pbp.empty or not required.issubset(pbp.columns): return pd.DataFrame()
    cols = ["game_id", "home_team", "away_team", "total_home_score", "total_away_score"]
    games = pbp[cols].copy()
    games["total_home_score"] = pd.to_numeric(games["total_home_score"], errors="coerce")
    games["total_away_score"] = pd.to_numeric(games["total_away_score"], errors="coerce")
    games = games.dropna(subset=["game_id", "home_team", "away_team", "total_home_score", "total_away_score"])
    if games.empty: return games
    return games.groupby("game_id", as_index=False).last()

def aggregate_team_epa(pbp):
    if pbp.empty:
        return {}, {"off_epa_per_play": 0.0, "off_pass_epa": 0.0, "off_rush_epa": 0.0, "def_epa_per_play": 0.0, "def_pass_epa": 0.0, "def_rush_epa": 0.0}, 0

    df = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    if df.empty: return {}, {}, 0
    df["epa"] = pd.to_numeric(df["epa"], errors="coerce")
    df = df.dropna(subset=["epa"])

    team_names = sorted(set(df["posteam"].dropna().unique()) | set(df["defteam"].dropna().unique()))
    league_off_epa = float(df["epa"].mean())
    stats_dict = {}

    for team in team_names:
        off = df[df["posteam"] == team]
        defense = df[df["defteam"] == team]
        pass_off = off[off["play_type"] == "pass"]
        rush_off = off[off["play_type"] == "run"]
        pass_def = defense[defense["play_type"] == "pass"]
        rush_def = defense[defense["play_type"] == "run"]

        stats_dict[team] = {
            "off_epa_per_play": float(off["epa"].mean()) if not off.empty else league_off_epa,
            "off_pass_epa": float(pass_off["epa"].mean()) if not pass_off.empty else league_off_epa,
            "off_rush_epa": float(rush_off["epa"].mean()) if not rush_off.empty else league_off_epa,
            "plays": int(len(off)),
            "def_plays": int(len(defense)),
            "pace": 63.0,
            "def_epa_per_play": float(defense["epa"].mean()) if not defense.empty else league_off_epa,
            "def_pass_epa": float(pass_def["epa"].mean()) if not pass_def.empty else league_off_epa,
            "def_rush_epa": float(rush_def["epa"].mean()) if not rush_def.empty else league_off_epa,
        }

    return stats_dict, {
        "off_epa_per_play": league_off_epa,
        "off_pass_epa": league_off_epa,
        "off_rush_epa": league_off_epa,
        "def_epa_per_play": league_off_epa,
        "def_pass_epa": league_off_epa,
        "def_rush_epa": league_off_epa,
    }, len(df)

def get_scoring_environment(pbp):
    games = get_final_game_scores(pbp)
    if games.empty: return 21.5, 0
    avg_points = float(pd.concat([games["total_home_score"], games["total_away_score"]], ignore_index=True).mean())
    return avg_points, int(len(games))

def fit_points_calibration(prior_pbp, prior_stats, prior_league_epa, prior_league_points):
    games = get_final_game_scores(prior_pbp)
    if games.empty or len(games) < 50:
        return {"epa_to_points_per_play": DEFAULT_EPA_TO_POINTS_PER_PLAY, "hfa_points": 1.8, "games_used": int(len(games)), "method": "fallback"}

    rows = []
    for _, game in games.iterrows():
        away = game["away_team"]
        home = game["home_team"]
        if away not in prior_stats or home not in prior_stats: continue

        away_off = prior_stats[away]["off_epa_per_play"]
        home_def = prior_stats[home]["def_epa_per_play"]
        home_off = prior_stats[home]["off_epa_per_play"]
        away_def = prior_stats[away]["def_epa_per_play"]

        # Proper match advantage logic
        away_matchup = (away_off - prior_league_epa) + (home_def - prior_league_epa)
        home_matchup = (home_off - prior_league_epa) + (away_def - prior_league_epa)

        away_points = safe_float(game["total_away_score"], np.nan)
        home_points = safe_float(game["total_home_score"], np.nan)
        if np.isnan(away_points) or np.isnan(home_points): continue

        rows.append({"matchup": away_matchup * 63.0, "home": 0.0, "points": away_points - prior_league_points})
        rows.append({"matchup": home_matchup * 63.0, "home": 1.0, "points": home_points - prior_league_points})

    if len(rows) < 100:
        return {"epa_to_points_per_play": DEFAULT_EPA_TO_POINTS_PER_PLAY, "hfa_points": 1.8, "games_used": len(rows) // 2, "method": "fallback"}

    calibration = pd.DataFrame(rows)
    X = calibration[["matchup", "home"]].to_numpy(dtype=float)
    y = calibration["points"].to_numpy(dtype=float)

    try:
        coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
        beta = float(coefficients[0])
        hfa = float(coefficients[1])
    except Exception:
        beta = DEFAULT_EPA_TO_POINTS_PER_PLAY
        hfa = 1.8

    beta = float(np.clip(beta, 0.25, 1.25))
    hfa = float(np.clip(hfa, 0.0, 4.0))

    return {"epa_to_points_per_play": beta, "hfa_points": hfa, "games_used": len(rows) // 2, "method": "2025 game-level least-squares calibration"}

def get_blended_nfl_stats(prior_season=PRIOR_SEASON, current_season=CURRENT_SEASON):
    print("Loading statistical baselines...")
    prior_raw = nfl.import_pbp_data([prior_season])
    prior_pbp = is_regular_season(prior_raw)

    try:
        current_raw = nfl.import_pbp_data([current_season])
        current_pbp = is_regular_season(current_raw)
    except Exception as exc:
        print(f"Current-season PBP unavailable: {exc}")
        current_pbp = pd.DataFrame()

    prior_stats, prior_league_stats, _ = aggregate_team_epa(prior_pbp)
    if current_pbp.empty:
        current_stats = {}
        current_league_stats = prior_league_stats
    else:
        current_stats, current_league_stats, _ = aggregate_team_epa(current_pbp)

    blended = {}
    all_teams = sorted(set(prior_stats) | set(current_stats))

    for team in all_teams:
        prior = prior_stats.get(team, {})
        current = current_stats.get(team, {})
        current_off_plays = int(current.get("plays", 0))
        current_def_plays = int(current.get("def_plays", 0))

        blended[team] = {
            "off_epa_per_play": weighted_current_value(current.get("off_epa_per_play"), prior.get("off_epa_per_play"), current_off_plays),
            "off_pass_epa": weighted_current_value(current.get("off_pass_epa"), prior.get("off_pass_epa"), current_off_plays),
            "off_rush_epa": weighted_current_value(current.get("off_rush_epa"), prior.get("off_rush_epa"), current_off_plays),
            "def_epa_per_play": weighted_current_value(current.get("def_epa_per_play"), prior.get("def_epa_per_play"), current_def_plays),
            "def_pass_epa": weighted_current_value(current.get("def_pass_epa"), prior.get("def_pass_epa"), current_def_plays),
            "def_rush_epa": weighted_current_value(current.get("def_rush_epa"), prior.get("def_rush_epa"), current_def_plays),
            "plays": current_off_plays or int(prior.get("plays", 0)),
            "def_plays": current_def_plays or int(prior.get("def_plays", 0)),
            "pace": float(current.get("pace", prior.get("pace", 63.0))),
        }

    prior_points, prior_games = get_scoring_environment(prior_pbp)
    current_points, current_games = get_scoring_environment(current_pbp)

    if current_games > 0:
        league_weight = current_games / (current_games + 32.0)
        league_points = (league_weight * current_points + (1.0 - league_weight) * prior_points)
    else:
        league_points = prior_points

    prior_epa = float(prior_league_stats.get("off_epa_per_play", 0.0))
    current_epa = float(current_league_stats.get("off_epa_per_play", prior_epa))

    if current_games > 0:
        epa_weight = current_games / (current_games + 32.0)
        league_epa = (epa_weight * current_epa + (1.0 - epa_weight) * prior_epa)
    else:
        league_epa = prior_epa

    calibration = fit_points_calibration(prior_pbp, prior_stats, prior_epa, prior_points)

    diagnostics = {
        "prior_season": prior_season, "current_season": current_season, "prior_games": prior_games,
        "current_games": current_games, "prior_league_points": round(prior_points, 3),
        "current_league_points": round(current_points, 3) if current_games else None,
        "blended_league_points": round(league_points, 3), "league_epa_baseline": round(league_epa, 6),
        "epa_prior_effective_plays": EPA_PRIOR_PLAYS, "calibration": calibration,
    }

    return blended, league_points, league_epa, diagnostics

def simulate_nfl_game(away_stats, home_stats, league_points=21.5, league_epa=0.0, hfa_points=1.8, epa_to_points=DEFAULT_EPA_TO_POINTS_PER_PLAY, total_line=45.0, spread_line=-3.0, num_sims=NUM_SIMS, rng=None):
    if rng is None: rng = np.random.default_rng(RANDOM_SEED)

    # CORRECTED SIGNS: Bad defense = positive EPA allowed = positive def_strength
    away_off_strength = away_stats.get("off_epa_per_play", 0.0) - league_epa
    away_def_strength = away_stats.get("def_epa_per_play", 0.0) - league_epa
    home_off_strength = home_stats.get("off_epa_per_play", 0.0) - league_epa
    home_def_strength = home_stats.get("def_epa_per_play", 0.0) - league_epa

    away_matchup_epa = away_off_strength + home_def_strength
    home_matchup_epa = home_off_strength + away_def_strength

    away_adjustment = float(np.clip(away_matchup_epa * 63.0 * epa_to_points, -MAX_MATCHUP_ADJUSTMENT_POINTS, MAX_MATCHUP_ADJUSTMENT_POINTS))
    home_adjustment = float(np.clip(home_matchup_epa * 63.0 * epa_to_points, -MAX_MATCHUP_ADJUSTMENT_POINTS, MAX_MATCHUP_ADJUSTMENT_POINTS))

    away_proj = max(3.0, float(league_points) + away_adjustment - hfa_points / 2.0)
    home_proj = max(3.0, float(league_points) + home_adjustment + hfa_points / 2.0)

    score_sd = DEFAULT_SCORE_SD
    rho = DEFAULT_SCORE_CORRELATION
    covariance = np.array([[score_sd ** 2, rho * score_sd ** 2], [rho * score_sd ** 2, score_sd ** 2]])
    draws = rng.multivariate_normal(mean=[away_proj, home_proj], cov=covariance, size=num_sims)

    away_sims = np.maximum(0, np.rint(draws[:, 0]).astype(int))
    home_sims = np.maximum(0, np.rint(draws[:, 1]).astype(int))

    ties = int(np.sum(away_sims == home_sims))
    away_wins = int(np.sum(away_sims > home_sims)) + ties / 2.0
    home_wins = int(np.sum(home_sims > away_sims)) + ties / 2.0
    
    total_sims = away_sims + home_sims
    margin_sims = home_sims - away_sims

    total_line_val = safe_float(total_line, 45.0)
    if np.isnan(total_line_val): total_line_val = 45.0
    spread_home = safe_float(spread_line, 0.0)
    if np.isnan(spread_home): spread_home = 0.0

    over_win = int(np.sum(total_sims > total_line_val))
    under_win = int(np.sum(total_sims < total_line_val))
    ou_push = int(np.sum(total_sims == total_line_val))

    away_cover = int(np.sum(margin_sims < -spread_home))
    home_cover = int(np.sum(margin_sims > -spread_home))
    spread_push = int(np.sum(margin_sims == -spread_home))

    return {
        "away_proj_score": round(away_proj, 1),
        "home_proj_score": round(home_proj, 1),
        "proj_total": round(away_proj + home_proj, 1),
        "fair_spread": round(away_proj - home_proj, 1),
        "fair_home_margin": round(home_proj - away_proj, 1),
        "away_win_prob": (away_wins / num_sims) * 100.0,
        "home_win_prob": (home_wins / num_sims) * 100.0,
        "ou_probs": {"over": over_win / num_sims, "under": under_win / num_sims, "push": ou_push / num_sims},
        "spread_probs": {"away": away_cover / num_sims, "home": home_cover / num_sims, "push": spread_push / num_sims},
        "diagnostics": {
            "away_off_strength": round(away_off_strength, 6), "away_def_strength": round(away_def_strength, 6),
            "home_off_strength": round(home_off_strength, 6), "home_def_strength": round(home_def_strength, 6),
            "away_matchup_epa": round(away_matchup_epa, 6), "home_matchup_epa": round(home_matchup_epa, 6),
            "away_adjustment": round(away_adjustment, 3), "home_adjustment": round(home_adjustment, 3),
            "epa_to_points_per_play": round(epa_to_points, 4), "hfa_points": round(hfa_points, 3),
        },
    }

def grade_historical_scores():
    if not os.path.exists("history.csv"): return
    print("Checking for ungraded games in history.csv...")
    df = pd.read_csv("history.csv")
    if df.empty: return
    
    is_ungraded = (df["Actual_Away_Score"].isna() | df["Actual_Away_Score"].astype(str).eq("N/A"))
    ungraded_dates = df[is_ungraded]["Date"].dropna().unique()
    if len(ungraded_dates) == 0:
        print("All games are graded.")
        return

    for target_date in ungraded_dates:
        try:
            dt = datetime.strptime(str(target_date), "%Y-%m-%d")
        except ValueError: continue
        
        check_dates = [dt.strftime("%Y%m%d"), (dt + pd.Timedelta(days=1)).strftime("%Y%m%d"), (dt - pd.Timedelta(days=1)).strftime("%Y%m%d")]
        scores = {}

        for dt_str in check_dates:
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={dt_str}"
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200: continue
                data = resp.json()
                for event in data.get("events", []):
                    for comp in event.get("competitions", []):
                        if not comp.get("status", {}).get("type", {}).get("completed", False): continue
                        for team in comp.get("competitors", []):
                            abbr = ESPN_TEAM_MAPPING.get(team.get("team", {}).get("name", ""))
                            if abbr: scores[f"{abbr}_{team.get('homeAway')}"] = team.get("score", 0)
            except Exception: continue

        for idx, row in df.iterrows():
            if not (pd.isna(row["Actual_Away_Score"]) or str(row["Actual_Away_Score"]) == "N/A"): continue
            away, home = row["Away_Team"], row["Home_Team"]
            if f"{away}_away" in scores and f"{home}_home" in scores:
                df.at[idx, "Actual_Away_Score"] = scores[f"{away}_away"]
                df.at[idx, "Actual_Home_Score"] = scores[f"{home}_home"]
                print(f"Graded: {away} {scores[f'{away}_away']} @ {home} {scores[f'{home}_home']}")

    df.to_csv("history.csv", index=False)
    print("history.csv grading pass complete.")

def run_live_scraper():
    print("Running Live Odds Scraper & Building Dashboard JSON...")
    if not ODDS_API_KEY: return

    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds?regions={REGIONS}&markets={MARKETS}&bookmakers={BOOKMAKER}&oddsFormat=american&apiKey={ODDS_API_KEY}"
    try: response = requests.get(url, timeout=20)
    except requests.RequestException: return
    if response.status_code != 200: return

    games = response.json()
    epa_stats, league_points, league_epa, diagnostics = get_blended_nfl_stats(PRIOR_SEASON, CURRENT_SEASON)

    history_file = "history.csv"
    history_fields = ["Date", "Away_Team", "Home_Team", "Pinnacle_Away_ML", "Pinnacle_Home_ML", "Pinnacle_Away_Spread", "Pinnacle_Home_Spread", "Pinnacle_Total_Line", "Actual_Away_Score", "Actual_Home_Score"]
    if not os.path.exists(history_file):
        with open(history_file, "w", newline="") as f: csv.writer(f).writerow(history_fields)
    
    try: existing_history = pd.read_csv(history_file)
    except Exception: existing_history = pd.DataFrame(columns=history_fields)

    dashboard_games = []
    rng = np.random.default_rng(RANDOM_SEED)

    for game in games:
        away = ODDS_API_TO_ABBR.get(game.get("away_team"))
        home = ODDS_API_TO_ABBR.get(game.get("home_team"))
        if not away or not home or away not in epa_stats or home not in epa_stats: continue

        try:
            commence_utc = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
            date_str = commence_utc.astimezone(ZoneInfo(DISPLAY_TIMEZONE)).strftime("%Y-%m-%d")
        except Exception:
            date_str = datetime.utcnow().strftime("%Y-%m-%d")

        pinnacle = next((b for b in game.get("bookmakers", []) if str(b.get("key", "")).lower() == BOOKMAKER.lower()), None)
        if pinnacle is None: continue
        markets = pinnacle.get("markets", [])

        away_ml = home_ml = "N/A"
        h2h_market = next((m for m in markets if m.get("key") == "h2h"), None)
        if h2h_market:
            for out in h2h_market.get("outcomes", []):
                if out.get("name") == game.get("away_team"): away_ml = out.get("price", "N/A")
                elif out.get("name") == game.get("home_team"): home_ml = out.get("price", "N/A")

        away_sp = home_sp = away_sp_odds = home_sp_odds = "N/A"
        spread_market = next((m for m in markets if m.get("key") == "spreads"), None)
        if spread_market:
            for out in spread_market.get("outcomes", []):
                if out.get("name") == game.get("away_team"): away_sp, away_sp_odds = out.get("point", "N/A"), out.get("price", "N/A")
                elif out.get("name") == game.get("home_team"): home_sp, home_sp_odds = out.get("point", "N/A"), out.get("price", "N/A")

        total_line = over_odds = under_odds = "N/A"
        totals_market = next((m for m in markets if m.get("key") == "totals"), None)
        if totals_market:
            for out in totals_market.get("outcomes", []):
                if out.get("name") == "Over": total_line, over_odds = out.get("point", "N/A"), out.get("price", "N/A")
                elif out.get("name") == "Under": under_odds = out.get("price", "N/A")

        # FIXED DEDUPLICATION: Ensures we do not write multiple entries for the same date regardless of grading status
        match_exists = not existing_history[
            (existing_history["Away_Team"] == away) & 
            (existing_history["Home_Team"] == home) &
            (existing_history["Date"] == date_str)
        ].empty

        if not match_exists and (away_ml != "N/A" or away_sp != "N/A" or total_line != "N/A"):
            with open(history_file, "a", newline="") as f:
                csv.writer(f).writerow([date_str, away, home, away_ml, home_ml, away_sp, home_sp, total_line, "N/A", "N/A"])

        sim_res = simulate_nfl_game(epa_stats[away], epa_stats[home], league_points=league_points, league_epa=league_epa, hfa_points=diagnostics["calibration"]["hfa_points"], epa_to_points=diagnostics["calibration"]["epa_to_points_per_play"], total_line=total_line, spread_line=home_sp, num_sims=NUM_SIMS, rng=rng)

        away_ml_ev = home_ml_ev = away_sp_ev = home_sp_ev = over_ev = under_ev = None
        if away_ml != "N/A" and home_ml != "N/A":
            try:
                t_away, t_home = proportional_devig(float(away_ml), float(home_ml))
                b_away = ((1.0 - ML_MARKET_WEIGHT) * (sim_res["away_win_prob"] / 100.0)) + (ML_MARKET_WEIGHT * t_away)
                b_home = ((1.0 - ML_MARKET_WEIGHT) * (sim_res["home_win_prob"] / 100.0)) + (ML_MARKET_WEIGHT * t_home)
                away_ml_ev = calculate_ev(b_away * 100.0, float(away_ml))
                home_ml_ev = calculate_ev(b_home * 100.0, float(home_ml))
            except: pass

        if away_sp_odds != "N/A" and home_sp_odds != "N/A":
            try:
                t_sp_a, t_sp_h = proportional_devig(float(away_sp_odds), float(home_sp_odds))
                b_sp_a = ((1.0 - SPREAD_MARKET_WEIGHT) * sim_res["spread_probs"]["away"]) + (SPREAD_MARKET_WEIGHT * t_sp_a)
                b_sp_h = ((1.0 - SPREAD_MARKET_WEIGHT) * sim_res["spread_probs"]["home"]) + (SPREAD_MARKET_WEIGHT * t_sp_h)
                b_sp_push = sim_res["spread_probs"]["push"]
                away_sp_ev = calculate_ev(b_sp_a * 100.0, float(away_sp_odds), b_sp_push * 100.0)
                home_sp_ev = calculate_ev(b_sp_h * 100.0, float(home_sp_odds), b_sp_push * 100.0)
            except: pass

        if over_odds != "N/A" and under_odds != "N/A":
            try:
                t_ou_o, t_ou_u = proportional_devig(float(over_odds), float(under_odds))
                b_ou_o = ((1.0 - TOTAL_MARKET_WEIGHT) * sim_res["ou_probs"]["over"]) + (TOTAL_MARKET_WEIGHT * t_ou_o)
                b_ou_u = ((1.0 - TOTAL_MARKET_WEIGHT) * sim_res["ou_probs"]["under"]) + (TOTAL_MARKET_WEIGHT * t_ou_u)
                b_ou_push = sim_res["ou_probs"]["push"]
                over_ev = calculate_ev(b_ou_o * 100.0, float(over_odds), b_ou_push * 100.0)
                under_ev = calculate_ev(b_ou_u * 100.0, float(under_odds), b_ou_push * 100.0)
            except: pass

        dashboard_games.append({
            "away_team": away, "home_team": home, "date": date_str, "target_date": date_str, "commence_time": game.get("commence_time"),
            "away_stats": epa_stats.get(away, {}), "home_stats": epa_stats.get(home, {}),
            "simulation": {
                "away_win_prob": round(sim_res["away_win_prob"], 1), "home_win_prob": round(sim_res["home_win_prob"], 1),
                "away_proj": sim_res["away_proj_score"], "home_proj": sim_res["home_proj_score"], "total_proj": sim_res["proj_total"],
                "fair_spread": sim_res["fair_spread"], "fair_home_margin": sim_res["fair_home_margin"],
                "fair_away_ml": prob_to_american(sim_res["away_win_prob"]), "fair_home_ml": prob_to_american(sim_res["home_win_prob"])
            },
            "market_data": {
                "h2h": {"away": away_ml, "home": home_ml}, "spreads": {"away": away_sp_odds, "home": home_sp_odds, "away_line": away_sp, "home_line": home_sp}, "totals": {"over": over_odds, "under": under_odds, "total": total_line},
                "away_ml_ev": round(away_ml_ev, 1) if away_ml_ev is not None else None,
                "home_ml_ev": round(home_ml_ev, 1) if home_ml_ev is not None else None,
                "away_sp_ev": round(away_sp_ev, 1) if away_sp_ev is not None else None,
                "home_sp_ev": round(home_sp_ev, 1) if home_sp_ev is not None else None,
                "over_ev": round(over_ev, 1) if over_ev is not None else None,
                "under_ev": round(under_ev, 1) if under_ev is not None else None,
            },
        })

    primary_date = dashboard_games[0]["date"] if dashboard_games else datetime.now(ZoneInfo(DISPLAY_TIMEZONE)).strftime("%Y-%m-%d")
    output_json = {
        "last_updated": datetime.now(ZoneInfo("UTC")).isoformat(), "slate": primary_date, "target_date": primary_date,
        "team_stats": epa_stats, "todays_games": dashboard_games, "games": dashboard_games,
    }
    with open("data.json", "w") as f: json.dump(output_json, f, indent=4)
    print(f"Scraping, JSON export, and line updates complete. Exported {len(dashboard_games)} games.")

if __name__ == "__main__":
    grade_historical_scores()
    run_live_scraper()
