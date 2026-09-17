import os
import json
import csv
import requests
import numpy as np
import pandas as pd
import scipy.stats as stats
import nfl_data_py as nfl
from datetime import datetime, timedelta

# NOTE: no global np.random.seed() anymore — each game now gets its own
# deterministic-but-independent RNG derived from the matchup, so simulating
# a full slate doesn't make each game's draw depend on how many games were
# simulated before it in the loop (see get_game_rng()).

ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '')
SPORT = 'americanfootball_nfl'
REGIONS = 'us'
MARKETS = 'h2h,spreads,totals'
BOOKMAKER = 'pinnacle'

LEAGUE_AVG_POINTS = 21.5
LEAGUE_AVG_HFA = 1.8
LEAGUE_AVG_STD = 9.5
LEAGUE_AVG_PACE = 63.0

# Manual starter-QB / key-injury override table.
# Populate before running the scraper on weeks where a starting QB (or other
# high-impact player) is confirmed OUT. epa_penalty is subtracted from that
# team's net offensive EPA for the game. Rough starting point: a backup QB
# swap is commonly worth -0.08 to -0.15 EPA/play; tune from your own tracking.
# Example: MANUAL_INJURY_OVERRIDES[('SF', 'SEA')] = {'team': 'HOME', 'epa_penalty': -0.10}
MANUAL_INJURY_OVERRIDES = {}

ODDS_API_TO_ABBR = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
}

ESPN_TEAM_MAPPING = {
    'Cardinals': 'ARI', 'Falcons': 'ATL', 'Ravens': 'BAL', 'Bills': 'BUF',
    'Panthers': 'CAR', 'Bears': 'CHI', 'Bengals': 'CIN', 'Browns': 'CLE',
    'Cowboys': 'DAL', 'Broncos': 'DEN', 'Lions': 'DET', 'Packers': 'GB',
    'Texans': 'HOU', 'Colts': 'IND', 'Jaguars': 'JAX', 'Chiefs': 'KC',
    'Raiders': 'LV', 'Chargers': 'LAC', 'Rams': 'LAR', 'Dolphins': 'MIA',
    'Vikings': 'MIN', 'Patriots': 'NE', 'Saints': 'NO', 'Giants': 'NYG',
    'Jets': 'NYJ', 'Eagles': 'PHI', 'Steelers': 'PIT', '49ers': 'SF',
    'Seahawks': 'SEA', 'Buccaneers': 'TB', 'Titans': 'TEN', 'Commanders': 'WAS'
}


def get_game_rng(away, home, date_str):
    """Per-game RNG seeded from the matchup+date, not a single global seed.
    This makes each game's draw independent of slate order/composition while
    staying reproducible if you re-run the exact same slate."""
    seed = abs(hash(f"{away}_{home}_{date_str}")) % (2**32)
    return np.random.default_rng(seed)


def opponent_adjust_epa(pbp_df, base_stats, iterations=2):
    """SRS-style iterative opponent adjustment. Raw EPA rewards teams that
    played a weak schedule and punishes teams that played a tough one; this
    subtracts out the average strength of opponents faced/allowed and
    re-centers on league average, repeating a couple of passes so the
    adjustment itself converges (an opponent's adjusted rating depends on
    who THEY played, etc.)."""
    if pbp_df.empty:
        return base_stats

    plays = pbp_df[(pbp_df['play_type'].isin(['pass', 'run'])) & pbp_df['epa'].notna()]
    matchups = plays[['game_id', 'posteam', 'defteam']].drop_duplicates()

    off_adj = {t: base_stats[t]['off_epa'] for t in base_stats}
    def_adj = {t: base_stats[t]['def_epa'] for t in base_stats}
    league_off_avg = float(np.mean(list(off_adj.values()))) if off_adj else 0.0
    league_def_avg = float(np.mean(list(def_adj.values()))) if def_adj else 0.0

    for _ in range(iterations):
        new_off, new_def = {}, {}
        for t in base_stats:
            opp_defs = matchups.loc[matchups['posteam'] == t, 'defteam'].unique()
            opp_def_strength = (np.mean([def_adj.get(o, league_def_avg) for o in opp_defs])
                                 if len(opp_defs) else league_def_avg)
            new_off[t] = base_stats[t]['off_epa'] - (opp_def_strength - league_def_avg)

            opp_offs = matchups.loc[matchups['defteam'] == t, 'posteam'].unique()
            opp_off_strength = (np.mean([off_adj.get(o, league_off_avg) for o in opp_offs])
                                 if len(opp_offs) else league_off_avg)
            new_def[t] = base_stats[t]['def_epa'] - (opp_off_strength - league_off_avg)
        off_adj, def_adj = new_off, new_def

    return {t: {'off_epa': off_adj[t], 'def_epa': def_adj[t]} for t in base_stats}


def compute_team_score_std(schedule_df, teams):
    """Team-specific scoring standard deviation instead of one league-wide
    constant. Falls back to the league average when a team has too few
    graded games to trust its own number."""
    std_dict = {}
    if schedule_df.empty:
        return {t: LEAGUE_AVG_STD for t in teams}
    for t in teams:
        scores = pd.concat([
            schedule_df.loc[schedule_df['home_team'] == t, 'home_score'],
            schedule_df.loc[schedule_df['away_team'] == t, 'away_score']
        ])
        team_std = scores.std()
        if len(scores) >= 6 and pd.notna(team_std) and team_std > 0:
            # blend toward league average to avoid small-sample noise
            std_dict[t] = 0.7 * team_std + 0.3 * LEAGUE_AVG_STD
        else:
            std_dict[t] = LEAGUE_AVG_STD
    return std_dict


def compute_team_hfa(schedule_df, teams):
    """Team-specific home field advantage from that team's own home/away
    scoring margin split, blended with the league average so a handful of
    games doesn't produce an extreme number."""
    hfa_dict = {}
    if schedule_df.empty:
        return {t: LEAGUE_AVG_HFA for t in teams}
    for t in teams:
        home_games = schedule_df[schedule_df['home_team'] == t]
        away_games = schedule_df[schedule_df['away_team'] == t]
        if len(home_games) >= 4 and len(away_games) >= 4:
            home_margin = (home_games['home_score'] - home_games['away_score']).mean()
            away_margin = (away_games['away_score'] - away_games['home_score']).mean()
            team_hfa = (home_margin - away_margin) / 2.0
            hfa_dict[t] = 0.5 * team_hfa + 0.5 * LEAGUE_AVG_HFA
        else:
            hfa_dict[t] = LEAGUE_AVG_HFA
    return hfa_dict


def compute_team_pace(pbp_df, teams):
    """Team-specific offensive plays/game instead of a fixed league number —
    pace varies enough team to team that this alone shifts total projections."""
    if pbp_df.empty:
        return {t: LEAGUE_AVG_PACE for t in teams}
    plays = pbp_df[pbp_df['play_type'].isin(['pass', 'run'])]
    games_played = plays.groupby('posteam')['game_id'].nunique()
    total_plays = plays.groupby('posteam').size()
    pace_dict = {}
    for t in teams:
        if t in games_played and games_played[t] > 0:
            pace_dict[t] = total_plays[t] / games_played[t]
        else:
            pace_dict[t] = LEAGUE_AVG_PACE
    return pace_dict


def get_blended_nfl_stats(prior_season=2025, current_season=2026):
    print("Loading statistical baselines...")
    pbp_prior = nfl.import_pbp_data([prior_season])
    try:
        pbp_curr = nfl.import_pbp_data([current_season])
    except Exception:
        pbp_curr = pd.DataFrame()

    def agg_epa(df):
        if df.empty:
            return {}
        df = df[(df['play_type'].isin(['pass', 'run'])) & (df['epa'].notna())]
        team_off_epa = df.groupby('posteam')['epa'].mean().to_dict()
        team_def_epa = df.groupby('defteam')['epa'].mean().to_dict()
        stats_dict = {}
        for t in team_off_epa.keys():
            stats_dict[t] = {
                'off_epa': team_off_epa.get(t, 0.0),
                'def_epa': team_def_epa.get(t, 0.0)
            }
        return stats_dict

    prior_stats = agg_epa(pbp_prior)
    curr_stats = agg_epa(pbp_curr)

    # Smooth recency weighting instead of a hard 1000-row cutoff: current
    # season phases in game-by-game per team (fully weighted by week 8)
    # rather than jumping from 0% to 80% the moment the league crosses a
    # row-count threshold.
    if not pbp_curr.empty:
        games_played_this_season = pbp_curr.groupby('posteam')['game_id'].nunique()
    else:
        games_played_this_season = pd.Series(dtype=float)

    blended = {}
    for t in prior_stats.keys():
        games_n = games_played_this_season.get(t, 0)
        weight_curr = min(1.0, games_n / 8.0)
        if t in curr_stats and weight_curr > 0:
            blended[t] = {
                'off_epa': weight_curr * curr_stats[t]['off_epa'] + (1 - weight_curr) * prior_stats[t]['off_epa'],
                'def_epa': weight_curr * curr_stats[t]['def_epa'] + (1 - weight_curr) * prior_stats[t]['def_epa']
            }
        else:
            blended[t] = prior_stats[t]

    # Opponent-adjust (SRS-style) using whichever pbp set is more populated
    adj_pbp = pbp_curr if not pbp_curr.empty else pbp_prior
    blended = opponent_adjust_epa(adj_pbp, blended, iterations=2)

    # Team-specific HFA / variance from recent schedule results
    try:
        schedule_df = nfl.import_schedules([prior_season, current_season])
        schedule_df = schedule_df.dropna(subset=['home_score', 'away_score'])
    except Exception:
        schedule_df = pd.DataFrame(columns=['home_team', 'away_team', 'home_score', 'away_score'])

    teams = list(blended.keys())
    team_std = compute_team_score_std(schedule_df, teams)
    team_hfa = compute_team_hfa(schedule_df, teams)
    team_pace = compute_team_pace(adj_pbp, teams)

    return blended, team_std, team_hfa, team_pace


def simulate_nfl_game(away_team, home_team, away_stats, home_stats,
                       away_std, home_std, hfa_points, away_pace, home_pace,
                       total_line=45.0, spread_line=-3.0, num_sims=10000,
                       injury_override=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    combined_pace = (away_pace + home_pace) / 2.0

    away_epa_net = away_stats['off_epa'] + home_stats['def_epa']
    home_epa_net = home_stats['off_epa'] + away_stats['def_epa']

    # Manual starter-QB / key-injury adjustment, e.g.
    # {'team': 'AWAY', 'epa_penalty': -0.10}
    if injury_override:
        if injury_override.get('team') == 'AWAY':
            away_epa_net += injury_override.get('epa_penalty', 0.0)
        elif injury_override.get('team') == 'HOME':
            home_epa_net += injury_override.get('epa_penalty', 0.0)

    away_proj = max(7.0, LEAGUE_AVG_POINTS + (away_epa_net * combined_pace) - (hfa_points / 2.0))
    home_proj = max(7.0, LEAGUE_AVG_POINTS + (home_epa_net * combined_pace) + (hfa_points / 2.0))

    proj_total = away_proj + home_proj
    proj_margin = home_proj - away_proj

    # Draw total and margin jointly instead of drawing away/home scores
    # independently. Real NFL games have a mild negative correlation between
    # total and |margin| (blowouts often see garbage-time scoring that
    # inflates the total relative to how much the margin itself grows), so
    # independent home/away draws overstate total variance and understate
    # margin variance in lopsided games. This is a simplified approximation,
    # not a fitted covariance — tune `correlation` against your own
    # historical total/margin data if you want to refine it further.
    total_std = np.sqrt(away_std ** 2 + home_std ** 2) * 0.9
    margin_std = np.sqrt(away_std ** 2 + home_std ** 2) * 0.85
    correlation = -0.15
    cov = correlation * total_std * margin_std
    cov_matrix = [[total_std ** 2, cov], [cov, margin_std ** 2]]

    draws = rng.multivariate_normal([proj_total, proj_margin], cov_matrix, num_sims)
    total_draw = draws[:, 0]
    margin_draw = draws[:, 1]

    home_sims = np.clip(np.round((total_draw + margin_draw) / 2.0), 0, None)
    away_sims = np.clip(np.round((total_draw - margin_draw) / 2.0), 0, None)

    total_sims = away_sims + home_sims
    margin_sims = home_sims - away_sims

    ties = np.sum(away_sims == home_sims)
    away_wins = np.sum(away_sims > home_sims) + (ties / 2.0)
    home_wins = np.sum(home_sims > away_sims) + (ties / 2.0)

    spread_home = spread_line if spread_line is not None else 0.0

    over_win = np.sum(total_sims > total_line) if total_line else 0
    under_win = np.sum(total_sims < total_line) if total_line else 0
    ou_push = np.sum(total_sims == total_line) if total_line else 0

    away_cover = np.sum(margin_sims < -spread_home) if spread_home is not None else 0
    home_cover = np.sum(margin_sims > -spread_home) if spread_home is not None else 0
    spread_push = np.sum(margin_sims == -spread_home) if spread_home is not None else 0

    return {
        "away_win_prob": (away_wins / num_sims) * 100.0,
        "home_win_prob": (home_wins / num_sims) * 100.0,
        "ou_probs": {
            "over": over_win / num_sims if total_line else 0.0,
            "under": under_win / num_sims if total_line else 0.0,
            "push": ou_push / num_sims if total_line else 0.0
        },
        "spread_probs": {
            "away": away_cover / num_sims if spread_home is not None else 0.0,
            "home": home_cover / num_sims if spread_home is not None else 0.0,
            "push": spread_push / num_sims if spread_home is not None else 0.0
        }
    }


def american_to_decimal(odds):
    if odds > 0:
        return 1.0 + (odds / 100.0)
    else:
        return 1.0 + (100.0 / abs(odds))


def implied_prob(odds):
    """Raw vig-included implied probability from American odds."""
    return 1.0 / american_to_decimal(odds)


def proportional_devig(odds_a, odds_b):
    """Two-way proportional devig: normalize both sides' implied
    probabilities so they sum to 1. For a two-outcome market this is
    identical to multiplicative devig."""
    p_a = implied_prob(odds_a)
    p_b = implied_prob(odds_b)
    total = p_a + p_b
    if total <= 0:
        return 0.5, 0.5
    return p_a / total, p_b / total


def calculate_ev(win_prob_pct, odds, push_prob_pct=0.0):
    """Expected value (%) of a bet given a win probability (0-100), American
    odds, and an optional push probability (0-100) that's excluded from the
    loss side."""
    if win_prob_pct is None or odds is None:
        return None
    win_prob = win_prob_pct / 100.0
    push_prob = (push_prob_pct or 0.0) / 100.0
    lose_prob = max(0.0, 1.0 - win_prob - push_prob)
    decimal_odds = american_to_decimal(odds)
    ev = (win_prob * (decimal_odds - 1.0)) - lose_prob
    return ev * 100.0


def grade_historical_scores():
    if not os.path.exists('history.csv'):
        return
    print("Checking for ungraded games in history.csv...")

    df = pd.read_csv('history.csv')
    updated = False

    is_ungraded = df['Actual_Away_Score'].isna() | (df['Actual_Away_Score'] == 'N/A')
    ungraded_dates = df[is_ungraded]['Date'].unique()

    if len(ungraded_dates) == 0:
        print("All games are graded.")
        return

    for target_date in ungraded_dates:
        try:
            dt = datetime.strptime(str(target_date), "%Y-%m-%d")
        except ValueError:
            continue

        check_dates = [
            dt.strftime("%Y%m%d"),
            (dt + timedelta(days=1)).strftime("%Y%m%d"),
            (dt - timedelta(days=1)).strftime("%Y%m%d")
        ]

        scores = {}
        for dt_str in check_dates:
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={dt_str}"
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    continue
                data = resp.json()

                for event in data.get('events', []):
                    for comp in event.get('competitions', []):
                        if comp.get('status', {}).get('type', {}).get('completed', False):
                            for team in comp.get('competitors', []):
                                mascot = team.get('team', {}).get('name', '')
                                score = team.get('score', 0)
                                home_away = team.get('homeAway')
                                abbr = ESPN_TEAM_MAPPING.get(mascot)
                                if abbr:
                                    scores[f"{abbr}_{home_away}"] = score
            except Exception:
                continue

        for idx, row in df.iterrows():
            if pd.isna(row['Actual_Away_Score']) or row['Actual_Away_Score'] == 'N/A':
                away = row['Away_Team']
                home = row['Home_Team']

                if f"{away}_away" in scores and f"{home}_home" in scores:
                    df.at[idx, 'Actual_Away_Score'] = scores[f"{away}_away"]
                    df.at[idx, 'Actual_Home_Score'] = scores[f"{home}_home"]
                    updated = True
                    print(f"Graded: {away} {scores[f'{away}_away']} @ {home} {scores[f'{home}_home']}")

    if updated:
        df.to_csv('history.csv', index=False)
        print("history.csv updated successfully.")


def run_live_scraper():
    print("Running Live Odds Scraper...")
    if not ODDS_API_KEY:
        print("Missing ODDS_API_KEY environment variable.")
        return

    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds?regions={REGIONS}&markets={MARKETS}&bookmakers={BOOKMAKER}&oddsFormat=american&apiKey={ODDS_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        print("Failed to fetch odds.")
        return

    games = response.json()
    epa_stats, team_std, team_hfa, team_pace = get_blended_nfl_stats(2025, 2026)

    history_file = 'history.csv'
    history_fields = ['Date', 'Away_Team', 'Home_Team', 'Pinnacle_Away_ML', 'Pinnacle_Home_ML', 'Pinnacle_Away_Spread', 'Pinnacle_Home_Spread', 'Pinnacle_Total_Line', 'Actual_Away_Score', 'Actual_Home_Score']

    if not os.path.exists(history_file):
        with open(history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(history_fields)

    existing_history = pd.read_csv(history_file)

    for game in games:
        away = ODDS_API_TO_ABBR.get(game['away_team'])
        home = ODDS_API_TO_ABBR.get(game['home_team'])
        if not away or not home or away not in epa_stats or home not in epa_stats:
            continue

        utc_time = datetime.strptime(game['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
        local_time = utc_time - timedelta(hours=5)
        date = local_time.strftime('%Y-%m-%d')

        bookmaker = game.get('bookmakers', [])
        if not bookmaker:
            continue
        markets = bookmaker[0].get('markets', [])

        h2h_market = next((m for m in markets if m['key'] == 'h2h'), None)
        spread_market = next((m for m in markets if m['key'] == 'spreads'), None)
        totals_market = next((m for m in markets if m['key'] == 'totals'), None)

        away_ml, home_ml = 'N/A', 'N/A'
        if h2h_market:
            for out in h2h_market['outcomes']:
                if out['name'] == game['away_team']:
                    away_ml = out['price']
                elif out['name'] == game['home_team']:
                    home_ml = out['price']

        away_sp, home_sp = 'N/A', 'N/A'
        if spread_market:
            for out in spread_market['outcomes']:
                if out['name'] == game['away_team']:
                    away_sp = out['point']
                elif out['name'] == game['home_team']:
                    home_sp = out['point']

        total_line = 'N/A'
        if totals_market:
            for out in totals_market['outcomes']:
                total_line = out['point']

        if away_ml == 'N/A' or away_sp == 'N/A' or total_line == 'N/A':
            continue

        is_ungraded = existing_history['Actual_Away_Score'].isna() | (existing_history['Actual_Away_Score'] == 'N/A')
        match_exists = not existing_history[(existing_history['Away_Team'] == away) & (existing_history['Home_Team'] == home) & is_ungraded].empty

        if not match_exists:
            with open(history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([date, away, home, away_ml, home_ml, away_sp, home_sp, total_line, 'N/A', 'N/A'])

        # --- Simulation preview using the enhanced model ---
        injury_override = MANUAL_INJURY_OVERRIDES.get((away, home))
        game_rng = get_game_rng(away, home, date)
        sim_result = simulate_nfl_game(
            away, home,
            epa_stats[away], epa_stats[home],
            team_std.get(away, LEAGUE_AVG_STD), team_std.get(home, LEAGUE_AVG_STD),
            team_hfa.get(home, LEAGUE_AVG_HFA),
            team_pace.get(away, LEAGUE_AVG_PACE), team_pace.get(home, LEAGUE_AVG_PACE),
            total_line=total_line if total_line != 'N/A' else 45.0,
            spread_line=home_sp if home_sp != 'N/A' else -3.0,
            rng=game_rng
        )
        print(f"{away} @ {home} ({date}): "
              f"Away win {sim_result['away_win_prob']:.1f}% / Home win {sim_result['home_win_prob']:.1f}% | "
              f"Over {sim_result['ou_probs']['over']*100:.1f}% / Under {sim_result['ou_probs']['under']*100:.1f}%")

    print("Scraping and line updates complete.")


if __name__ == '__main__':
    grade_historical_scores()
    run_live_scraper()
