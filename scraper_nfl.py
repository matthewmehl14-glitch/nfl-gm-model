import os
import requests
import json
import csv
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
    """Safely fetch play-by-play data, filtering out garbage time (WP < 5% or > 95%)."""
    try:
        pbp = nfl.import_pbp_data([season])
        if pbp is not None and not pbp.empty:
            pbp = pbp[(pbp['play_type'].isin(['pass', 'run'])) & (pbp['epa'].notna())]
            # Garbage time filter
            if 'wp' in pbp.columns:
                pbp = pbp[(pbp['wp'] >= 0.05) & (pbp['wp'] <= 0.95)]
            return pbp
    except Exception as e:
        print(f"Notice: No data available for {season} ({e}).")
    return None

def compute_raw_stats(pbp):
    """Blends EPA (60%) and Success Rate (40%) across a 75/25 Pass/Rush split."""
    if pbp is None or pbp.empty:
        return {}

    pbp = pbp.copy()
    pbp['success'] = (pbp['epa'] > 0).astype(int)

    off_stats = pbp.groupby(['posteam', 'play_type']).agg(epa=('epa', 'mean'), sr=('success', 'mean')).reset_index()
    def_stats = pbp.groupby(['defteam', 'play_type']).agg(epa=('epa', 'mean'), sr=('success', 'mean')).reset_index()
    
    vol = pbp.groupby('posteam').agg(plays=('play_id', 'count'), games=('game_id', 'nunique')).reset_index()

    stats = {}
    for _, row in vol.iterrows():
        team = row['posteam']
        t_off = off_stats[off_stats['posteam'] == team]
        pass_off = t_off[t_off['play_type'] == 'pass']
        run_off = t_off[t_off['play_type'] == 'run']
        
        off_pass_epa = pass_off['epa'].values[0] if len(pass_off) > 0 else 0.0
        off_rush_epa = run_off['epa'].values[0] if len(run_off) > 0 else 0.0
        off_pass_sr = pass_off['sr'].values[0] if len(pass_off) > 0 else 0.0
        off_rush_sr = run_off['sr'].values[0] if len(run_off) > 0 else 0.0
        
        # 75/25 Pass/Rush Split
        comp_off_epa = (0.75 * off_pass_epa) + (0.25 * off_rush_epa)
        comp_off_sr = (0.75 * off_pass_sr) + (0.25 * off_rush_sr)
        
        # Drive Quality (60% EPA, 40% SR relative to NFL avg of 0.44)
        off_dq = (0.60 * comp_off_epa) + (0.40 * (comp_off_sr - 0.44))

        stats[team] = {
            "off_epa_per_play": float(off_dq),
            "off_pass_epa": float(off_pass_epa),
            "off_rush_epa": float(off_rush_epa),
            "plays": int(row['plays']),
            "pace": float(row['plays'] / row['games']) if row['games'] > 0 else 63.0
        }

    for team in stats.keys():
        t_def = def_stats[def_stats['defteam'] == team]
        pass_def = t_def[t_def['play_type'] == 'pass']
        run_def = t_def[t_def['play_type'] == 'run']
        
        def_pass_epa = pass_def['epa'].values[0] if len(pass_def) > 0 else 0.0
        def_rush_epa = run_def['epa'].values[0] if len(run_def) > 0 else 0.0
        def_pass_sr = pass_def['sr'].values[0] if len(pass_def) > 0 else 0.0
        def_rush_sr = run_def['sr'].values[0] if len(run_def) > 0 else 0.0
        
        comp_def_epa = (0.75 * def_pass_epa) + (0.25 * def_rush_epa)
        comp_def_sr = (0.75 * def_pass_sr) + (0.25 * def_rush_sr)
        
        def_dq = (0.60 * comp_def_epa) + (0.40 * (comp_def_sr - 0.44))
        
        stats[team].update({
            "def_epa_per_play": float(def_dq),
            "def_pass_epa": float(def_pass_epa),
            "def_rush_epa": float(def_rush_epa)
        })

    return stats

def get_blended_nfl_stats(prior_season=2025, current_season=2026, sample_threshold=400.0):
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
        w_current = min(1.0, current_plays / sample_threshold)
        w_prior = 1.0 - w_current

        metrics = ["off_epa_per_play", "off_pass_epa", "off_rush_epa", "def_epa_per_play", "def_pass_epa", "def_rush_epa", "pace"]

        blended_stats[team] = {"plays": current_plays}
        for m in metrics:
            val_prior = p_team.get(m, 0.0)
            val_current = c_team.get(m, val_prior)
            blended_stats[team][m] = round((w_current * val_current) + (w_prior * val_prior), 3)

    return blended_stats

def get_pinnacle_odds(api_key):
    if not api_key: return {}
    cutoff = (datetime.utcnow() + timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ')
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

def proportional_devig(odds_a, odds_b):
    if not odds_a or not odds_b:
        return 0.5, 0.5
    dec_a = american_to_decimal(odds_a)
    dec_b = american_to_decimal(odds_b)
    imp_a = 1.0 / dec_a
    imp_b = 1.0 / dec_b
    total_imp = imp_a + imp_b
    if total_imp == 0: return 0.5, 0.5
    return imp_a / total_imp, imp_b / total_imp

def calculate_ev(prob_pct, am_odds, push_pct=0.0):
    if not am_odds or prob_pct == 0: return None
    prob, p_push, dec = prob_pct / 100.0, push_pct / 100.0, american_to_decimal(am_odds)
    return round(((prob * dec) - 1.0 + p_push) * 100, 1)

def calc_kelly_units(prob_pct, am_odds, push_pct=0.0, multiplier=0.25):
    if not am_odds or prob_pct == 0: return 0.0
    prob = prob_pct / 100.0
    b = american_to_decimal(am_odds) - 1.0
    q = 1.0 - prob
    k = ((b * prob) - q) / b
    if k > 0:
        units = round((k * 100) * multiplier, 2)
        return min(units, 2.0)
    return 0.0

def simulate_nfl_game(away_epa, home_epa, total_line=None, spread_line=None, iterations=10000):
    expected_plays = (away_epa.get("pace", 63.0) + home_epa.get("pace", 63.0)) / 2.0
    scale_factor = 0.55
    away_net_epa = (away_epa["off_epa_per_play"] + home_epa["def_epa_per_play"]) * scale_factor
    home_net_epa = (home_epa["off_epa_per_play"] + away_epa["def_epa_per_play"]) * scale_factor
    
    away_exp = max(13.0, min(34.0, 21.5 + (away_net_epa * expected_plays)))
    home_exp = max(13.0, min(34.0, 23.0 + (home_net_epa * expected_plays)))
    
    away_sims = np.random.normal(away_exp, 9.5, iterations)
    home_sims = np.random.normal(home_exp, 9.5, iterations)
    
    margin_sims = np.round(home_sims - away_sims)
    total_sims = np.round(away_sims + home_sims)
    
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
        "away_win_prob": float(away_win / iterations) * 100,
        "home_win_prob": float(home_win / iterations) * 100,
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
        
    if total_line is not None:
        over = np.sum(total_sims > total_line)
        under = np.sum(total_sims < total_line)
        push_tot = np.sum(total_sims == total_line)
        results["ou_probs"] = {
            "over": float(over / iterations),
            "under": float(under / iterations),
            "push": float(push_tot / iterations)
        }
        
    return results

def update_history_csv(todays_games, date_str):
    file_name = 'history.csv'
    headers = [
        'Date', 'Away_Team', 'Home_Team', 'Away_Win_Prob', 'Home_Win_Prob',
        'Away_Proj_Score', 'Home_Proj_Score', 'Proj_Total',
        'Pinnacle_Away_ML', 'Pinnacle_Home_ML', 'Pinnacle_Away_Spread', 'Pinnacle_Home_Spread', 'Pinnacle_Total_Line',
        'Away_ML_EV', 'Home_ML_EV', 'Away_Sp_EV', 'Home_Sp_EV', 'Over_EV', 'Under_EV',
        'Actual_Away_Score', 'Actual_Home_Score', 'Actual_Total'
    ]
    
    existing_data = {}
    if os.path.exists(file_name):
        with open(file_name, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['Date']}_{row['Away_Team']}_{row['Home_Team']}"
                existing_data[key] = row
                
    for game in todays_games:
        sim = game['simulation']
        mkt = game['market_data']
        key = f"{date_str}_{game['away_team']}_{game['home_team']}"
        
        actual_away = existing_data.get(key, {}).get('Actual_Away_Score', 'N/A')
        actual_home = existing_data.get(key, {}).get('Actual_Home_Score', 'N/A')
        actual_total = existing_data.get(key, {}).get('Actual_Total', 'N/A')

        existing_data[key] = {
            'Date': date_str,
            'Away_Team': game['away_team'],
            'Home_Team': game['home_team'],
            'Away_Win_Prob': sim.get('away_win_prob', 'N/A'),
            'Home_Win_Prob': sim.get('home_win_prob', 'N/A'),
            'Away_Proj_Score': sim.get('away_proj', 'N/A'),
            'Home_Proj_Score': sim.get('home_proj', 'N/A'),
            'Proj_Total': sim.get('total_proj', 'N/A'),
            'Pinnacle_Away_ML': mkt.get('h2h', {}).get('away', 'N/A'),
            'Pinnacle_Home_ML': mkt.get('h2h', {}).get('home', 'N/A'),
            'Pinnacle_Away_Spread': mkt.get('spreads', {}).get('away_line', 'N/A'),
            'Pinnacle_Home_Spread': mkt.get('spreads', {}).get('home_line', 'N/A'),
            'Pinnacle_Total_Line': mkt.get('totals', {}).get('point', 'N/A'),
            'Away_ML_EV': mkt.get('away_ml_ev', 'N/A'),
            'Home_ML_EV': mkt.get('home_ml_ev', 'N/A'),
            'Away_Sp_EV': mkt.get('away_sp_ev', 'N/A'),
            'Home_Sp_EV': mkt.get('home_sp_ev', 'N/A'),
            'Over_EV': mkt.get('over_ev', 'N/A'),
            'Under_EV': mkt.get('under_ev', 'N/A'),
            'Actual_Away_Score': actual_away,
            'Actual_Home_Score': actual_home,
            'Actual_Total': actual_total
        }
        
    try:
        unique_dates = set([row['Date'].replace("-", "") for row in existing_data.values() if row.get('Actual_Away_Score') in ['N/A', '', None]])
        
        for u_date in unique_dates:
            espn_url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={u_date}"
            res = requests.get(espn_url)
            if res.status_code == 200:
                espn_data = res.json()
                for event in espn_data.get('events', []):
                    competition = event['competitions'][0]
                    status = competition['status']['type']
                    if status['completed']:
                        competitors = competition['competitors']
                        team1 = competitors[0]
                        team2 = competitors[1]
                        
                        t1_abbr = team1['team']['abbreviation']
                        t2_abbr = team2['team']['abbreviation']
                        t1_abbr = "LA" if t1_abbr == "LAR" else ("WAS" if t1_abbr == "WSH" else t1_abbr)
                        t2_abbr = "LA" if t2_abbr == "LAR" else ("WAS" if t2_abbr == "WSH" else t2_abbr)
                        
                        for row in existing_data.values():
                            if row['Date'].replace("-", "") == u_date and row.get('Actual_Away_Score') in ['N/A', '', None]:
                                if (row['Away_Team'] == t1_abbr and row['Home_Team'] == t2_abbr) or (row['Away_Team'] == t2_abbr and row['Home_Team'] == t1_abbr):
                                    if team1['homeAway'] == 'home':
                                        home_s = int(team1['score'])
                                        away_s = int(team2['score'])
                                    else:
                                        away_s = int(team1['score'])
                                        home_s = int(team2['score'])
                                        
                                    row['Actual_Away_Score'] = away_s
                                    row['Actual_Home_Score'] = home_s
                                    row['Actual_Total'] = away_s + home_s
    except Exception as e:
        print(f"Could not auto-fetch ESPN final scores: {e}")
        
    with open(file_name, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in existing_data.values():
            writer.writerow(row)

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
                model_weight = 0.50
                
                # Shrink Moneyline Probabilities using Proportional De-vigging
                away_odds = mkt.get('h2h', {}).get('away')
                home_odds = mkt.get('h2h', {}).get('home')
                t_away, t_home = proportional_devig(away_odds, home_odds)
                
                if t_away != 0.5:
                    sim_res["away_win_prob"] = (model_weight * (sim_res["away_win_prob"] / 100.0)) + ((1.0 - model_weight) * t_away)
                    sim_res["home_win_prob"] = (model_weight * (sim_res["home_win_prob"] / 100.0)) + ((1.0 - model_weight) * t_home)
                else:
                    sim_res["away_win_prob"] = sim_res["away_win_prob"] / 100.0
                    sim_res["home_win_prob"] = sim_res["home_win_prob"] / 100.0
                    
                # Shrink Spread Probabilities
                sp_away_odds = mkt.get('spreads', {}).get('away')
                sp_home_odds = mkt.get('spreads', {}).get('home')
                t_sp_away, t_sp_home = proportional_devig(sp_away_odds, sp_home_odds)
                if t_sp_away != 0.5:
                    sim_res["spread_probs"]["away"] = (model_weight * sim_res["spread_probs"]["away"]) + ((1.0 - model_weight) * t_sp_away)
                    sim_res["spread_probs"]["home"] = (model_weight * sim_res["spread_probs"]["home"]) + ((1.0 - model_weight) * t_sp_home)

                # Shrink Totals Probabilities
                ou_over_odds = mkt.get('totals', {}).get('over')
                ou_under_odds = mkt.get('totals', {}).get('under')
                t_over, t_under = proportional_devig(ou_over_odds, ou_under_odds)
                if t_over != 0.5:
                    sim_res["ou_probs"]["over"] = (model_weight * sim_res["ou_probs"]["over"]) + ((1.0 - model_weight) * t_over)
                    sim_res["ou_probs"]["under"] = (model_weight * sim_res["ou_probs"]["under"]) + ((1.0 - model_weight) * t_under)

                # Calculate EV & Units
                mkt["away_ml_ev"] = calculate_ev(sim_res["away_win_prob"] * 100, away_odds)
                mkt["away_ml_units"] = calc_kelly_units(sim_res["away_win_prob"] * 100, away_odds)
                mkt["home_ml_ev"] = calculate_ev(sim_res["home_win_prob"] * 100, home_odds)
                mkt["home_ml_units"] = calc_kelly_units(sim_res["home_win_prob"] * 100, home_odds)
                
                # Format final win prob display
                sim_res["away_win_prob"] = round(sim_res["away_win_prob"] * 100, 1)
                sim_res["home_win_prob"] = round(sim_res["home_win_prob"] * 100, 1)
                
                if spread_line is not None:
                    mkt["away_sp_ev"] = calculate_ev(sim_res["spread_probs"]["away"] * 100, sp_away_odds, sim_res["spread_probs"]["push"] * 100)
                    mkt["away_sp_units"] = calc_kelly_units(sim_res["spread_probs"]["away"] * 100, sp_away_odds, sim_res["spread_probs"]["push"] * 100)
                    mkt["home_sp_ev"] = calculate_ev(sim_res["spread_probs"]["home"] * 100, sp_home_odds, sim_res["spread_probs"]["push"] * 100)
                    mkt["home_sp_units"] = calc_kelly_units(sim_res["spread_probs"]["home"] * 100, sp_home_odds, sim_res["spread_probs"]["push"] * 100)
                
                if total_line is not None:
                    mkt["over_ev"] = calculate_ev(sim_res["ou_probs"]["over"] * 100, ou_over_odds, sim_res["ou_probs"]["push"] * 100)
                    mkt["over_units"] = calc_kelly_units(sim_res["ou_probs"]["over"] * 100, ou_over_odds, sim_res["ou_probs"]["push"] * 100)
                    mkt["under_ev"] = calculate_ev(sim_res["ou_probs"]["under"] * 100, ou_under_odds, sim_res["ou_probs"]["push"] * 100)
                    mkt["under_units"] = calc_kelly_units(sim_res["ou_probs"]["under"] * 100, ou_under_odds, sim_res["ou_probs"]["push"] * 100)

            todays_games.append({
                "away_team": away_abbr, "home_team": home_abbr,
                "away_offense": epa_stats[away_abbr], "home_offense": epa_stats[home_abbr],
                "simulation": sim_res, "market_data": mkt
            })

    date_str = datetime.now().strftime('%Y-%m-%d')
    output_data = {
        "date": date_str,
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "teams": epa_stats,
        "todays_games": todays_games
    }
    
    with open('data.json', 'w') as f:
        json.dump(output_data, f, indent=4)
        
    update_history_csv(todays_games, date_str)
    print(f"NFL Engine successfully updated for {len(todays_games)} matchups.")

if __name__ == "__main__":
    generate_nfl_json()
