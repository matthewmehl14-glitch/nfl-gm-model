import csv
import numpy as np
from scraper_nfl import (
    get_blended_nfl_stats,
    simulate_nfl_game,
    proportional_devig,
    calculate_ev,
    american_to_decimal,
    safe_float,
    ML_MARKET_WEIGHT,
    SPREAD_MARKET_WEIGHT,
    TOTAL_MARKET_WEIGHT
)

# Seed for consistent simulations
np.random.seed(42)

def evaluate_bet(market_type, pick, row, bet_amount=25.0):
    away_score = safe_float(row.get('Actual_Away_Score'))
    home_score = safe_float(row.get('Actual_Home_Score'))
    actual_total = away_score + home_score

    if market_type == 'ML':
        odds = safe_float(row.get('Pinnacle_Away_ML') if pick == 'away' else row.get('Pinnacle_Home_ML'))
        won = (away_score > home_score) if pick == 'away' else (home_score > away_score)
        if away_score == home_score:
            return 0.0, 'push'
        profit = bet_amount * (american_to_decimal(odds) - 1.0) if won else -bet_amount
        return profit, ('win' if won else 'loss')

    elif market_type == 'SPREAD':
        line = safe_float(row.get('Pinnacle_Away_Spread') if pick == 'away' else row.get('Pinnacle_Home_Spread'))
        margin = (away_score - home_score) if pick == 'away' else (home_score - away_score)
        if margin + line == 0:
            return 0.0, 'push'
        won = (margin + line) > 0
        profit = bet_amount * (100 / 110) if won else -bet_amount  
        return profit, ('win' if won else 'loss')

    elif market_type == 'TOTAL':
        total_line = safe_float(row.get('Pinnacle_Total_Line'))
        if actual_total == total_line:
            return 0.0, 'push'
        won = (actual_total > total_line) if pick == 'over' else (actual_total < total_line)
        profit = bet_amount * (100 / 110) if won else -bet_amount
        return profit, ('win' if won else 'loss')

def run_backtest(target_date=None, flat_bet=25.0):
    # Tiered EV Hurdles
    ml_min, ml_max = 3.0, 10.0
    sp_min, sp_max = 1.5, 7.0
    tot_min, tot_max = 1.5, 7.0
    
    # Properly unpack the 4 variables from the new engine
    epa_stats, league_points, league_epa, diagnostics = get_blended_nfl_stats(2025, 2026)

    games_evaluated = 0
    total_staked = 0.0
    total_profit = 0.0
    results_summary = {
        'ML': {'W': 0, 'L': 0, 'P': 0},
        'SPREAD': {'W': 0, 'L': 0, 'P': 0},
        'TOTAL': {'W': 0, 'L': 0, 'P': 0}
    }

    with open('history.csv', mode='r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))

    print(f"\n--- REPLAYING SLATE WITH FLAT ${flat_bet:.2f} BETS (DECOUPLED EV CAPS) ---\n")

    for row in reader:
        if target_date and row['Date'] != target_date:
            continue
            
        away_score = safe_float(row.get('Actual_Away_Score'))
        home_score = safe_float(row.get('Actual_Home_Score'))
        
        # Skip if the game isn't finished yet
        if away_score is None or home_score is None:
            continue

        away = row['Away_Team']
        home = row['Home_Team']
        if away not in epa_stats or home not in epa_stats:
            continue

        games_evaluated += 1
        
        total_line = safe_float(row.get('Pinnacle_Total_Line'))
        spread_line = safe_float(row.get('Pinnacle_Home_Spread'))
        away_ml = safe_float(row.get('Pinnacle_Away_ML'))
        home_ml = safe_float(row.get('Pinnacle_Home_ML'))

        # Explicitly assign kwargs to map to the new dynamic calibrations
        sim_res = simulate_nfl_game(
            epa_stats[away], 
            epa_stats[home], 
            league_points=league_points,
            league_epa=league_epa,
            hfa_points=diagnostics["calibration"]["hfa_points"],
            epa_to_points=diagnostics["calibration"]["epa_to_points_per_play"],
            total_line=total_line, 
            spread_line=spread_line
        )

        # --- Moneyline Check ---
        if away_ml is not None and home_ml is not None:
            t_away, t_home = proportional_devig(away_ml, home_ml)
            
            # Extract 0-1 percentage from the 0-100 return format
            model_away = sim_res["away_win_prob"] / 100.0
            model_home = sim_res["home_win_prob"] / 100.0

            # New Engine weighting equation: (1 - Market_Wt) * Model + (Market_Wt * Market)
            b_away = ((1.0 - ML_MARKET_WEIGHT) * model_away) + (ML_MARKET_WEIGHT * t_away)
            b_home = ((1.0 - ML_MARKET_WEIGHT) * model_home) + (ML_MARKET_WEIGHT * t_home)

            away_ml_ev = calculate_ev(b_away * 100.0, away_ml)
            home_ml_ev = calculate_ev(b_home * 100.0, home_ml)

            if away_ml_ev and ml_min <= away_ml_ev <= ml_max:
                p, res = evaluate_bet('ML', 'away', row, flat_bet)
                total_staked += flat_bet
                total_profit += p
                results_summary['ML'][res[0].upper()] += 1
                print(f"Betted ML: {away} ({res.upper()}) | EV: {away_ml_ev:.1f}% | Stake: ${flat_bet:.2f} | Profit: ${p:.2f}")

            elif home_ml_ev and ml_min <= home_ml_ev <= ml_max:
                p, res = evaluate_bet('ML', 'home', row, flat_bet)
                total_staked += flat_bet
                total_profit += p
                results_summary['ML'][res[0].upper()] += 1
                print(f"Betted ML: {home} ({res.upper()}) | EV: {home_ml_ev:.1f}% | Stake: ${flat_bet:.2f} | Profit: ${p:.2f}")

        # --- Spread Check ---
        if spread_line is not None and safe_float(row.get('Pinnacle_Away_Spread')) is not None:
            t_sp_a, t_sp_h = proportional_devig(-110, -110)
            
            model_away_sp = sim_res["spread_probs"]["away"]
            model_home_sp = sim_res["spread_probs"]["home"]

            b_sp_a = ((1.0 - SPREAD_MARKET_WEIGHT) * model_away_sp) + (SPREAD_MARKET_WEIGHT * t_sp_a)
            b_sp_h = ((1.0 - SPREAD_MARKET_WEIGHT) * model_home_sp) + (SPREAD_MARKET_WEIGHT * t_sp_h)
            b_sp_push = sim_res["spread_probs"]["push"]

            away_sp_ev = calculate_ev(b_sp_a * 100.0, -110, b_sp_push * 100.0)
            home_sp_ev = calculate_ev(b_sp_h * 100.0, -110, b_sp_push * 100.0)

            if away_sp_ev and sp_min <= away_sp_ev <= sp_max:
                p, res = evaluate_bet('SPREAD', 'away', row, flat_bet)
                total_staked += flat_bet
                total_profit += p
                results_summary['SPREAD'][res[0].upper()] += 1
                print(f"Betted SPREAD: {away} ({res.upper()}) | EV: {away_sp_ev:.1f}% | Stake: ${flat_bet:.2f} | Profit: ${p:.2f}")

            elif home_sp_ev and sp_min <= home_sp_ev <= sp_max:
                p, res = evaluate_bet('SPREAD', 'home', row, flat_bet)
                total_staked += flat_bet
                total_profit += p
                results_summary['SPREAD'][res[0].upper()] += 1
                print(f"Betted SPREAD: {home} ({res.upper()}) | EV: {home_sp_ev:.1f}% | Stake: ${flat_bet:.2f} | Profit: ${p:.2f}")

        # --- Totals Check ---
        if total_line is not None:
            t_ou_o, t_ou_u = proportional_devig(-110, -110)
            
            model_over = sim_res["ou_probs"]["over"]
            model_under = sim_res["ou_probs"]["under"]

            b_ou_o = ((1.0 - TOTAL_MARKET_WEIGHT) * model_over) + (TOTAL_MARKET_WEIGHT * t_ou_o)
            b_ou_u = ((1.0 - TOTAL_MARKET_WEIGHT) * model_under) + (TOTAL_MARKET_WEIGHT * t_ou_u)
            b_ou_push = sim_res["ou_probs"]["push"]

            over_ev = calculate_ev(b_ou_o * 100.0, -110, b_ou_push * 100.0)
            under_ev = calculate_ev(b_ou_u * 100.0, -110, b_ou_push * 100.0)

            if over_ev and tot_min <= over_ev <= tot_max:
                p, res = evaluate_bet('TOTAL', 'over', row, flat_bet)
                total_staked += flat_bet
                total_profit += p
                results_summary['TOTAL'][res[0].upper()] += 1
                print(f"Betted TOTAL: OVER {total_line} in {away}@{home} ({res.upper()}) | EV: {over_ev:.1f}% | Stake: ${flat_bet:.2f} | Profit: ${p:.2f}")

            elif under_ev and tot_min <= under_ev <= tot_max:
                p, res = evaluate_bet('TOTAL', 'under', row, flat_bet)
                total_staked += flat_bet
                total_profit += p
                results_summary['TOTAL'][res[0].upper()] += 1
                print(f"Betted TOTAL: UNDER {total_line} in {away}@{home} ({res.upper()}) | EV: {under_ev:.1f}% | Stake: ${flat_bet:.2f} | Profit: ${p:.2f}")

    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    
    print("\n--- FINAL FLAT-BETTING BACKTEST RESULTS ---")
    print(f"Games Evaluated:  {games_evaluated}")
    print(f"Total Bets Placed: {sum(results_summary['ML'].values()) + sum(results_summary['SPREAD'].values()) + sum(results_summary['TOTAL'].values())}")
    print(f"Moneyline Record: {results_summary['ML']['W']}-{results_summary['ML']['L']}-{results_summary['ML']['P']}")
    print(f"Spread Record:    {results_summary['SPREAD']['W']}-{results_summary['SPREAD']['L']}-{results_summary['SPREAD']['P']}")
    print(f"Totals Record:    {results_summary['TOTAL']['W']}-{results_summary['TOTAL']['L']}-{results_summary['TOTAL']['P']}")
    print(f"Total Staked:     ${total_staked:.2f}")
    print(f"Net Profit:       ${total_profit:.2f}")
    print(f"Recalibrated ROI: {roi:.2f}%")

if __name__ == '__main__':
    run_backtest()
