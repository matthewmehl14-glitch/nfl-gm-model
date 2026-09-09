import os
import requests
import json
import numpy as np
import nfl_data_py as nfl
from datetime import datetime, timedelta

TEAM_MAPPING = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
}

def load_season_pbp(season):
    """Safely fetch play-by-play data for a single season."""
    try:
        pbp = nfl.import_pbp_data([season])
        if pbp is not None and not pbp.empty:
            pbp = pbp[(pbp['play_type'].isin(['pass', 'run'])) & (pbp['epa'].notna())]
            return pbp
    except Exception as e:
        print(f"Notice: No data available for {season} ({e}).")
    return None

def compute_raw_stats(pbp):
    """Calculates offensive/defensive EPA and pace from a play-by-play DataFrame."""
    if pbp is None or pbp.empty:
        return {}

    off_epa = pbp.groupby('posteam').agg(
        off_epa_per_play=('epa', 'mean'),
        off_pass_epa=('epa', lambda x: x[pbp['play_type'] == 'pass'].mean()),
        off_rush_epa=('epa', lambda x: x[pbp['play_type'] == 'run'].mean()),
        plays=('play_id', 'count'),
        games=('game_id', 'nunique')
    ).reset_index()

    def_epa = pbp.groupby('defteam').agg(
        def_epa_per_play=('epa', 'mean'),
        def_pass_epa=('epa', lambda x: x[pbp['play_type'] == 'pass'].mean()),
        def_rush_epa=('epa', lambda x: x[pbp['play_type'] == 'run'].mean())
    ).reset_index()

    stats = {}
    for _, row in off_epa.iterrows():
        stats[row['posteam']] = {
            "off_epa_per_play": float(row['off_epa_per_play']),
            "off_pass_epa": float(row['off_pass_epa']),
            "off_rush_epa": float(row['off_rush_epa']),
            "plays": int(row['plays']),
            "pace": float(row['plays'] / row['games']) if row['games'] > 0 else 63.0
        }

    for _, row in def_epa.iterrows():
        team = row['defteam']
        if team in stats:
            stats[team].update({
                "def_epa_per_play": float(row['def_epa_per_play']),
                "def_pass_epa": float(row['def_pass_epa']),
                "def_rush_epa": float(row['def_rush_epa'])
            })
    return stats

def get_blended_nfl_stats(prior_season=2025, current_season=2026, sample_threshold=400.0):
    """Blends prior season baselines with current season metrics using play-count weighting."""
    print(f"Loading {prior_season} baseline data...")
    pbp_prior = load_season_pbp(prior_season)
    prior_stats = compute_raw_stats(pbp_prior)

    print(f"Checking for {current_season} in-season data...")
    pbp_current = load_season_pbp(current_season)
    current_stats = compute_raw_stats(pbp_current)

    blended_stats = {}
    all_teams = set(prior_stats.keys()).union(set(current_stats.keys()))

    for team in all_teams:
        p_team = prior_stats.get(team, {})
        c_team = current_stats.get(team, {})

        current_plays = c_team.get("plays", 0)

        # Scale weight from 0.0 to 1.0 based on offensive snap volume
        w_current = min(1.0, current_plays / sample_threshold)
        w_prior = 1.0 - w_current

        metrics = [
            "off_epa_per_play", "off_pass_epa", "off_rush_epa",
            "def_epa_per_play", "def_pass_epa", "def_rush_epa", "pace"
        ]

        blended_stats[team] = {"plays": current_plays}
        for m in metrics:
            val_prior = p_team.get(m, 0.0)
            val_current = c_team.get(m, val_prior)
            blended_stats[team][m] = round((w_current * val_current) + (w_prior * val_prior), 3)

    return blended_stats

def get_pinnacle_odds(api_key):
    if not api_key:
        return {}
        
    # Calculate a strict 7-day cutoff window to isolate the current NFL week
    cutoff = (datetime.utcnow() + timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Append commenceTimeTo to the API URL
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey={api_key}&bookmakers=pinnacle&markets=h2h,spreads,totals&oddsFormat=american&commenceTimeTo={cutoff}"
    
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        odds_dict = {}
        for game in data:
            home, away = game.get('home_team'), game.get('away_team')
            if home in TEAM_MAPPING and away in TEAM_MAPPING:
                home_abbr, away_abbr = TEAM_MAPPING[home], TEAM_MAPPING[away]
                matchup_key = f"{away_abbr}@{home_abbr}"
                
                game_odds = {'h2h': {}, 'spreads': {}, 'totals': {}}
                for book in game.get('bookmakers', []):
                    if book['key'] == 'pinnacle':
                        for market in book.get('markets', []):
                            for out in market['outcomes']:
                                if market['key'] == 'h2h':
                                    if out['name'] == home: game_odds['h2h']['home'] = out['price']
                                    elif out['name'] == away: game_odds['h2h']['away'] = out['price']
                                elif market['key'] == 'spreads':
                                    if out['name'] == home: 
                                        game_odds['spreads']['home'] = out['price']
                                        game_odds['spreads']['home_line'] = out['point']
                                    elif out['name'] == away: 
                                        game_odds['spreads']['away'] = out['price']
                                        game_odds['spreads']['away_line'] = out['point']
                                elif market['key'] == 'totals':
                                    if out['name'] == 'Over':
                                        game_odds['totals']['over'] = out['price']
                                        game_odds['totals']['point'] = out.get('point')
                                    elif out['name'] == 'Under':
                                        game_odds['totals']['under'] = out['price']
                odds_dict[matchup_key] = game_odds
        return odds_dict
    except Exception as e:
        print(f"Odds API Error: {e}")
        return {}

def american_to_decimal(am_odds):
    return (am_odds / 100.0) + 1.0 if am_odds > 0 else (100.0 / abs(am_odds)) + 1.0

def calculate_ev(prob_pct, am_odds, push_pct=0.0):
    if not am_odds or prob_pct == 0:
        return None
    prob, p_push, dec = prob_pct / 100.0, push_pct / 100.0, american_to_decimal(am_odds)
    return round(((prob * dec) - 1.0 + p_push) * 100, 1)

def calc_kelly_units(prob_pct, am_odds, push_pct=0.0, multiplier=0.25):
    if not am_odds or prob_pct == 0:
        return 0.0
    
    prob = prob_pct / 100.0
    b = american_to_decimal(am_odds) - 1.0
    q = 1.0 - prob
    k = ((b * prob) - q) / b
    
    if k > 0:
        units = round((k * 100) * multiplier, 2)
        # Enforce a strict 2.0 unit ceiling to protect the bankroll
        return min(units, 2.0)
        
    return 0.0

def simulate_nfl_game(away_epa, home_epa, total_line=None, spread_line=None, iterations=10000):
    # Dynamic Pace Engine
    expected_plays = (away_epa.get("pace", 63.0) + home_epa.get("pace", 63.0)) / 2.0
    
    away_adv = (away_epa["off_epa_per_play"] - home_epa["def_epa_per_play"]) * expected_plays
    home_adv = (home_epa["off_epa_per_play"] - away_epa["def_epa_per_play"]) * expected_plays
    
    away_exp = 21.0 + away_adv
    home_exp = 22.5 + home_adv 
    
    away_sims = np.random.normal(away_exp, 9.5, iterations)
    home_sims = np.random.normal(home_exp, 9.5, iterations)
    
    margin_sims = np.round(home_sims - away_sims)
    total_sims = np.round(away_sims + home_sims)
    
    # Key Number Clustering (3, 7, 10 margins)
    for i in range(iterations):
        if margin_sims[i] in [2, 4] and np.random.random() < 0.35: margin_sims[i] = 3
        elif margin_sims[i] in [-2, -4] and np.random.random() < 0.35: margin_sims[i] = -3
        elif margin_sims[i] in [6, 8] and np.random.random() < 0.25: margin_sims[i] = 7
        elif margin_sims[i] in [-6, -8] and np.random.random() < 0.25: margin_sims[i] = -7
        elif margin_sims[i] in [9, 11] and np.random.random() < 0.20: margin_sims[i] = 10
        elif margin_sims[i] in [-9, -11] and np.random.random() < 0.20: margin_sims[i] = -10
    
    home_win = np.sum(margin_sims > 0)
    away_win = np.sum(margin_sims < 0)
    
    results = {
        "away_win_prob": round(float(away_win / iterations) * 100, 1),
        "home_win_prob": round(float(home_win / iterations) * 100, 1),
        "away_proj": round(float(np.mean(away_sims)), 1),
        "home_proj": round(float(np.mean(home_sims)), 1),
        "total_proj": round(float(np.mean(total_sims)), 1),
        "spread_probs": {"away": 0, "home": 0, "push": 0},
        "ou_probs": {"over": 0, "under": 0, "push": 0}
    }
    
    if spread_line is not None:
        home_cover = np.sum(margin_sims > spread_line * -1)
        away_cover = np.sum(margin_sims < spread_line * -1)
        push_spread = np.sum(margin_sims == spread_line * -1)
        results["spread_probs"] = {
            "away": float(away_cover / iterations),
            "home": float(home_cover / iterations),
            "push": float(push_spread / iterations)
        }
        
    if total_line:
        over = np.sum(total_sims > total_line)
        under = np.sum(total_sims < total_line)
        push_tot = np.sum(total_sims == total_line)
        results["ou_probs"] = {
            "over": float(over / iterations),
            "under": float(under / iterations),
            "push": float(push_tot / iterations)
        }
        
    return results

def generate_nfl_json():
    print("Executing dynamic 2025/2026 Bayesian EPA blend...")
    epa_stats = get_blended_nfl_stats(prior_season=2025, current_season=2026, sample_threshold=400.0)
    
    api_key = os.environ.get("ODDS_API_KEY")
    pinnacle_data = get_pinnacle_odds(api_key)
    todays_games = []

    for match_key, mkt in pinnacle_data.items():
        away_abbr, home_abbr = match_key.split('@')
        if away_abbr in epa_stats and home_abbr in epa_stats:
            total_line = mkt.get('totals', {}).get('point')
            spread_line = mkt.get('spreads', {}).get('home_line')
            
            sim_res = simulate_nfl_game(
                epa_stats[away_abbr], epa_stats[home_abbr], total_line, spread_line
            )
            
            if mkt:
                mkt["away_ml_ev"] = calculate_ev(sim_res["away_win_prob"], mkt.get('h2h', {}).get('away'))
                mkt["away_ml_units"] = calc_kelly_units(sim_res["away_win_prob"], mkt.get('h2h', {}).get('away'))
                mkt["home_ml_ev"] = calculate_ev(sim_res["home_win_prob"], mkt.get('h2h', {}).get('home'))
                mkt["home_ml_units"] = calc_kelly_units(sim_res["home_win_prob"], mkt.get('h2h', {}).get('home'))
                
                if spread_line is not None:
                    mkt["away_sp_ev"] = calculate_ev(sim_res["spread_probs"]["away"] * 100, mkt.get('spreads', {}).get('away'), sim_res["spread_probs"]["push"] * 100)
                    mkt["away_sp_units"] = calc_kelly_units(sim_res["spread_probs"]["away"] * 100, mkt.get('spreads', {}).get('away'), sim_res["spread_probs"]["push"] * 100)
                    mkt["home_sp_ev"] = calculate_ev(sim_res["spread_probs"]["home"] * 100, mkt.get('spreads', {}).get('home'), sim_res["spread_probs"]["push"] * 100)
                    mkt["home_sp_units"] = calc_kelly_units(sim_res["spread_probs"]["home"] * 100, mkt.get('spreads', {}).get('home'), sim_res["spread_probs"]["push"] * 100)
                
            todays_games.append({
                "away_team": away_abbr, "home_team": home_abbr,
                "away_offense": epa_stats[away_abbr], "home_offense": epa_stats[home_abbr],
                "simulation": sim_res, "market_data": mkt
            })

    output_data = {
        "date": datetime.now().strftime('%Y-%m-%d'),
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "teams": epa_stats,
        "todays_games": todays_games
    }
    with open('data.json', 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"NFL Engine successfully updated for {len(todays_games)} matchups.")

if __name__ == "__main__":
    generate_nfl_json()
