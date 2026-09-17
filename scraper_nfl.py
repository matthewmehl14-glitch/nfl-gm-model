import os
import json
import csv
import requests
import numpy as np
import pandas as pd
import scipy.stats as stats
import nfl_data_py as nfl
from datetime import datetime

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
                'def_epa': team_def_epa.get(t, 0.0)  # Positive = allows EPA (bad), Negative = stifles EPA (good)
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
    # Standard NFL game: ~63 offensive plays per team
    PLAYS_PER_GAME = 63.0
    LEAGUE_AVG_POINTS = 21.5
    HFA_POINTS = 1.8  # Home field advantage

    # Correct sign: Positive def_epa means defense allows more points
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
    
    ungraded_dates = df[(df['Actual_Away_Score'].isna()) | (df['Actual_Away_Score'] == 'N/A')]['Date'].unique()
    
    if len(ungraded_dates) == 0:
        print("All games are graded.")
        return

    for target_date in ungraded_dates:
        dt_str = str(target_date).replace('-', '')
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={dt_str}"
        
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200: continue
            data = resp.json()
        except Exception:
            continue
        
        scores = {}
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
        
        for idx, row in df[df['Date'] == target_date].iterrows():
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
    epa_stats = get_blended_nfl_stats(2025, 2026)
    
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
            
        date = game['commence_time'][:10]
        bookmaker = game.get('bookmakers', [])
        if not bookmaker: continue
        markets = bookmaker[0].get('markets', [])
        
        h2h_market = next((m for m in markets if m['key'] == 'h2h'), None)
        spread_market = next((m for m in markets if m['key'] == 'spreads'), None)
        totals_market = next((m for m in markets if m['key'] == 'totals'), None)
        
        away_ml, home_ml = 'N/A', 'N/A'
        if h2h_market:
            for out in h2h_market['outcomes']:
                if out['name'] == game['away_team']: away_ml = out['price']
                elif out['name'] == game['home_team']: home_ml = out['price']
                
        away_sp, home_sp = 'N/A', 'N/A'
        if spread_market:
            for out in spread_market['outcomes']:
                if out['name'] == game['away_team']: away_sp = out['point']
                elif out['name'] == game['home_team']: home_sp = out['point']
                    
        total_line = 'N/A'
        if totals_market:
            for out in totals_market['outcomes']:
                total_line = out['point']

        if away_ml == 'N/A' or away_sp == 'N/A' or total_line == 'N/A':
            continue
            
        match_exists = not existing_history.empty and not existing_history[(existing_history['Date'] == date) & (existing_history['Away_Team'] == away) & (existing_history['Home_Team'] == home)].empty
        
        if not match_exists:
            with open(history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([date, away, home, away_ml, home_ml, away_sp, home_sp, total_line, 'N/A', 'N/A'])

    print("Scraping and line updates complete.")

if __name__ == '__main__':
    grade_historical_scores()
    run_live_scraper()
