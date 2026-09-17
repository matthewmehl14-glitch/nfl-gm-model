import os
import json
import csv
import requests
import numpy as np
import pandas as pd
import scipy.stats as stats
import nfl_data_py as nfl
from datetime import datetime, timedelta

# Set seed for reproducible Monte Carlo draws
np.random.seed(42)

ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '')
SPORT = 'americanfootball_nfl'
REGIONS = 'us'
MARKETS = 'h2h,spreads,totals'
BOOKMAKER = 'pinnacle'

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

def get_blended_nfl_stats(prior_season=2025, current_season=2026):
    print("Loading statistical baselines...")
    pbp_prior = nfl.import_pbp_data([prior_season])
    try:
        pbp_curr = nfl.import_pbp_data([current_season])
    except Exception:
        pbp_curr = pd.DataFrame()

    def agg_epa(df):
        if df.empty: return {}
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

    blended = {}
    for t in prior_stats.keys():
        if t in curr_stats and pbp_curr.shape[0] > 1000:
            blended[t] = {
                'off_epa': 0.8 * curr_stats[t]['off_epa'] + 0.2 * prior_stats[t]['off_epa'],
                'def_epa': 0.8 * curr_stats[t]['def_epa'] + 0.2 * prior_stats[t]['def_epa']
            }
        else:
            blended[t] = prior_stats[t]
    return blended

def simulate_nfl_game(away_stats, home_stats, total_line=45.0, spread_line=-3.0, num_sims=10000):
    PLAYS_PER_GAME = 63.0
    LEAGUE_AVG_POINTS = 21.5
    HFA_POINTS = 1.8 

    away_epa_net = away_stats['off_epa'] + home_stats['def_epa']
    home_epa_net = home_stats['off_epa'] + away_stats['def_epa']

    away_proj = max(7.0, LEAGUE_AVG_POINTS + (away_epa_net * PLAYS_PER_GAME) - (HFA_POINTS / 2.0))
    home_proj = max(7.0, LEAGUE_AVG_POINTS + (home_epa_net * PLAYS_PER_GAME) + (HFA_POINTS / 2.0))

    away_sims = np.round(np.random.normal(away_proj, 9.5, num_sims)).astype(int)
    home_sims = np.round(np.random.normal(home_proj, 9.5, num_sims)).astype(int)

    ties = np.sum(away_sims == home_sims)
    away_wins = np.sum(away_sims > home_sims) + (ties / 2.0)
    home_wins = np.sum(home_sims > away_sims) + (ties / 2.0)
    
    total_sims = away_sims + home_sims
    margin_sims = home_sims - away_sims 

    spread_home = spread_line if spread_line is not None and spread_line != 'N/A' else 0.0
    total_line_val = total_line if total_line is not None and total_line != 'N/A' else 45.0
    
    over_win = np.sum(total_sims > total_line_val)
    under_win = np.sum(total_sims < total_line_val)
    ou_push = np.sum(total_sims == total_line_val)

    away_cover = np.sum(margin_sims < -spread_home)
    home_cover = np.sum(margin_sims > -spread_home)
    spread_push = np.sum(margin_sims == -spread_home)

    return {
        "away_proj_score": round(away_proj, 1),
        "home_proj_score": round(home_proj, 1),
        "proj_total": round(away_proj + home_proj, 1),
        "fair_spread": round(home_proj - away_proj, 1),
        "away_win_prob": (away_wins / num_sims) * 100.0,
        "home_win_prob": (home_wins / num_sims) * 100.0,
        "ou_probs": {
            "over": over_win / num_sims,
            "under": under_win / num_sims,
            "push": ou_push / num_sims
        },
        "spread_probs": {
            "away": away_cover / num_sims,
            "home": home_cover / num_sims,
            "push": spread_push / num_sims
        }
    }

def prob_to_american(p):
    if p <= 0: return "+0"
    if p >= 100: return "-99999"
    dec = 100.0 / p
    if dec >= 2.0: return f"+{int(round((dec - 1) * 100))}"
    else: return f"{int(round(-100 / (dec - 1)))}"

def american_to_decimal(odds):
    if odds > 0: return 1.0 + (odds / 100.0)
    else: return 1.0 + (100.0 / abs(odds))

def proportional_devig(odds1, odds2):
    p1 = 1.0 / american_to_decimal(odds1)
    p2 = 1.0 / american_to_decimal(odds2)
    total = p1 + p2
    return p1 / total, p2 / total

def calculate_ev(win_prob, odds, push_prob=0.0):
    dec_odds = american_to_decimal(odds)
    win_p = win_prob / 100.0
    push_p = push_prob / 100.0
    loss_p = max(0.0, 1.0 - win_p - push_p)
    profit_on_win = dec_odds - 1.0
    return ((win_p * profit_on_win) - loss_p) * 100.0

def calc_kelly_units(win_prob, odds, push_prob=0.0, fraction=0.25, max_unit=2.0):
    dec_odds = american_to_decimal(odds)
    win_p = win_prob / 100.0
    b = dec_odds - 1.0
    if b <= 0: return 0.0
    kelly_f = (win_p * b - (1.0 - win_p)) / b
    if kelly_f <= 0: return 0.0
    adj_kelly = kelly_f * fraction * 100.0 
    return round(min(adj_kelly, max_unit), 2)

def grade_historical_scores():
    if not os.path.exists('history.csv'): return
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
                if resp.status_code != 200: continue
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
    print("Running Live Odds Scraper & Building Dashboard JSON...")
    if not ODDS_API_KEY:
        print("Missing ODDS_API_KEY environment variable.")
        return
        
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds?regions={REGIONS}&markets={MARKETS}&bookmakers={BOOKMAKER}&oddsFormat=american&apiKey={ODDS_API_KEY}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Failed to fetch odds.")
        return
        
    games = response.json()
    epa_stats = get_blended_nfl_stats(2025, 2026)
    
    history_file = 'history.csv'
    history_fields = ['Date', 'Away_Team', 'Home_Team', 'Pinnacle_Away_ML', 'Pinnacle_Home_ML', 'Pinnacle_Away_Spread', 'Pinnacle_Home_Spread', 'Pinnacle_Total_Line', 'Actual_Away_Score', 'Actual_Home_Score']
    
    if not os.path.exists(history_file):
        with open(history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(history_fields)
            
    existing_history = pd.read_csv(history_file)
    dashboard_games = []
    
    ml_weight, sp_weight, tot_weight = 0.15, 0.40, 0.35
    ml_min, ml_max = 3.0, 10.0
    sp_min, sp_max = 1.5, 7.0
    tot_min, tot_max = 1.5, 7.0
    
    for game in games:
        away = ODDS_API_TO_ABBR.get(game['away_team'])
        home = ODDS_API_TO_ABBR.get(game['home_team'])
        if not away or not home or away not in epa_stats or home not in epa_stats:
            continue
            
        utc_time = datetime.strptime(game['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
        local_time = utc_time - timedelta(hours=5) 
        date_str = local_time.strftime('%Y-%m-%d')
        
        bookmaker = game.get('bookmakers', [])
        if not bookmaker: continue
        markets = bookmaker[0].get('markets', [])
        
        # Parse ML
        away_ml, home_ml = 'N/A', 'N/A'
        h2h_market = next((m for m in markets if m['key'] == 'h2h'), None)
        if h2h_market:
            for out in h2h_market['outcomes']:
                if out['name'] == game['away_team']: away_ml = out['price']
                elif out['name'] == game['home_team']: home_ml = out['price']
                
        # Parse Spread
        away_sp, home_sp, away_sp_odds, home_sp_odds = 'N/A', 'N/A', 'N/A', 'N/A'
        spread_market = next((m for m in markets if m['key'] == 'spreads'), None)
        if spread_market:
            for out in spread_market['outcomes']:
                if out['name'] == game['away_team']: 
                    away_sp, away_sp_odds = out['point'], out['price']
                elif out['name'] == game['home_team']: 
                    home_sp, home_sp_odds = out['point'], out['price']
                    
        # Parse Totals
        total_line, over_odds, under_odds = 'N/A', 'N/A', 'N/A'
        totals_market = next((m for m in markets if m['key'] == 'totals'), None)
        if totals_market:
            for out in totals_market['outcomes']:
                if out['name'] == 'Over':
                    total_line, over_odds = out['point'], out['price']
                elif out['name'] == 'Under':
                    under_odds = out['price']
            
        # CSV Deduplication Logic
        is_ungraded = existing_history['Actual_Away_Score'].isna() | (existing_history['Actual_Away_Score'] == 'N/A')
        match_exists = not existing_history[(existing_history['Away_Team'] == away) & (existing_history['Home_Team'] == home) & is_ungraded].empty
        
        if not match_exists and (away_ml != 'N/A' or away_sp != 'N/A' or total_line != 'N/A'):
            with open(history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([date_str, away, home, away_ml, home_ml, away_sp, home_sp, total_line, 'N/A', 'N/A'])

        # --- GENERATE DASHBOARD DATA (ALLOWS MISSING LINES) ---
        sim_res = simulate_nfl_game(epa_stats[away], epa_stats[home], total_line, home_sp)
        
        # Moneyline Recs
        away_ml_ev, home_ml_ev, away_ml_rec, home_ml_rec = None, None, None, None
        if away_ml != 'N/A' and home_ml != 'N/A':
            t_away, t_home = proportional_devig(away_ml, home_ml)
            b_away = (ml_weight * (sim_res["away_win_prob"] / 100.0)) + ((1.0 - ml_weight) * t_away)
            b_home = (ml_weight * (sim_res["home_win_prob"] / 100.0)) + ((1.0 - ml_weight) * t_home)
            ev_a = calculate_ev(b_away * 100, away_ml)
            ev_h = calculate_ev(b_home * 100, home_ml)
            if ml_min <= ev_a <= ml_max: 
                away_ml_ev = ev_a
                away_ml_rec = calc_kelly_units(b_away * 100, away_ml)
            if ml_min <= ev_h <= ml_max: 
                home_ml_ev = ev_h
                home_ml_rec = calc_kelly_units(b_home * 100, home_ml)

        # Spread Recs
        away_sp_ev, home_sp_ev, away_sp_rec, home_sp_rec = None, None, None, None
        if away_sp_odds != 'N/A' and home_sp_odds != 'N/A':
            t_sp_a, t_sp_h = proportional_devig(away_sp_odds, home_sp_odds)
            b_sp_a = (sp_weight * sim_res["spread_probs"]["away"]) + ((1.0 - sp_weight) * t_sp_a)
            b_sp_h = (sp_weight * sim_res["spread_probs"]["home"]) + ((1.0 - sp_weight) * t_sp_h)
            b_sp_push = sim_res["spread_probs"]["push"]
            ev_sp_a = calculate_ev(b_sp_a * 100, away_sp_odds, b_sp_push * 100)
            ev_sp_h = calculate_ev(b_sp_h * 100, home_sp_odds, b_sp_push * 100)
            if sp_min <= ev_sp_a <= sp_max:
                away_sp_ev = ev_sp_a
                away_sp_rec = calc_kelly_units(b_sp_a * 100, away_sp_odds, b_sp_push * 100)
            if sp_min <= ev_sp_h <= sp_max:
                home_sp_ev = ev_sp_h
                home_sp_rec = calc_kelly_units(b_sp_h * 100, home_sp_odds, b_sp_push * 100)

        # Total Recs
        over_ev, under_ev, over_rec, under_rec = None, None, None, None
        if over_odds != 'N/A' and under_odds != 'N/A':
            t_ou_o, t_ou_u = proportional_devig(over_odds, under_odds)
            b_ou_o = (tot_weight * sim_res["ou_probs"]["over"]) + ((1.0 - tot_weight) * t_ou_o)
            b_ou_u = (tot_weight * sim_res["ou_probs"]["under"]) + ((1.0 - tot_weight) * t_ou_u)
            b_ou_push = sim_res["ou_probs"]["push"]
            ev_o = calculate_ev(b_ou_o * 100, over_odds, b_ou_push * 100)
            ev_u = calculate_ev(b_ou_u * 100, under_odds, b_ou_push * 100)
            if tot_min <= ev_o <= tot_max:
                over_ev = ev_o
                over_rec = calc_kelly_units(b_ou_o * 100, over_odds, b_ou_push * 100)
            if tot_min <= ev_u <= tot_max:
                under_ev = ev_u
                under_rec = calc_kelly_units(b_ou_u * 100, under_odds, b_ou_push * 100)

        dashboard_games.append({
            "id": f"{away}_{home}",
            "matchup": f"{away} @ {home}",
            "Date": date_str,
            "date": date_str,
            "Away_Team": away,
            "Home_Team": home,
            "AwayTeam": away,
            "HomeTeam": home,
            "away_team": away,
            "home_team": home,
            "away_team_full": game['away_team'],
            "home_team_full": game['home_team'],
            "commence_time": game['commence_time'],
            
            "away_prob": round(sim_res["away_win_prob"], 1),
            "home_prob": round(sim_res["home_win_prob"], 1),
            "proj_away_score": sim_res["away_proj_score"],
            "proj_home_score": sim_res["home_proj_score"],
            "proj_total": sim_res["proj_total"],
            "fair_spread": sim_res["fair_spread"],
            "fair_away_ml": prob_to_american(sim_res["away_win_prob"]),
            "fair_home_ml": prob_to_american(sim_res["home_win_prob"]),
            
            "away_ml": away_ml,
            "home_ml": home_ml,
            "away_ml_ev": round(away_ml_ev, 1) if away_ml_ev is not None else None,
            "home_ml_ev": round(home_ml_ev, 1) if home_ml_ev is not None else None,
            "away_ml_rec": away_ml_rec if away_ml_rec is not None else None,
            "home_ml_rec": home_ml_rec if home_ml_rec is not None else None,
            
            "away_spread": away_sp,
            "home_spread": home_sp,
            "away_spread_odds": away_sp_odds,
            "home_spread_odds": home_sp_odds,
            "away_spread_ev": round(away_sp_ev, 1) if away_sp_ev is not None else None,
            "home_spread_ev": round(home_sp_ev, 1) if home_sp_ev is not None else None,
            "away_spread_rec": away_sp_rec if away_sp_rec is not None else None,
            "home_spread_rec": home_sp_rec if home_sp_rec is not None else None,
            
            "total_line": total_line,
            "over_odds": over_odds,
            "under_odds": under_odds,
            "over_ev": round(over_ev, 1) if over_ev is not None else None,
            "under_ev": round(under_ev, 1) if under_ev is not None else None,
            "over_rec": over_rec if over_rec is not None else None,
            "under_rec": under_rec if under_rec is not None else None
        })

    primary_date = dashboard_games[0]['date'] if dashboard_games else (datetime.utcnow() - timedelta(hours=5)).strftime('%Y-%m-%d')
    
    output_json = {
        "last_updated": datetime.utcnow().isoformat() + "Z", 
        "slate": primary_date,
        "slate_date": primary_date,
        "date": primary_date,
        "games": dashboard_games,
        "matches": dashboard_games,
        "matchups": dashboard_games
    }
    with open('data.json', 'w') as f:
        json.dump(output_json, f, indent=4)

    print(f"Scraping, JSON export, and line updates complete. Exported {len(dashboard_games)} games.")

if __name__ == '__main__':
    grade_historical_scores()
    run_live_scraper()
