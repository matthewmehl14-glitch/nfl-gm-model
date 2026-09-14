import csv
import numpy as np
from scraper_nfl import (
    get_blended_nfl_stats,
    simulate_nfl_game,
    proportional_devig,
    calculate_ev,
    calc_kelly_units,
    american_to_decimal
)

def evaluate_bet(market_type, pick, row, bet_size=25.0):
    away_score = float(row['Actual_Away_Score'])
    home_score = float(row['Actual_Home_Score'])
    actual_total = away_score + home_score

    if market_type == 'ML':
        odds = float(row['Pinnacle_Away_ML'] if pick == 'away' else row['Pinnacle_Home_ML'])
        won = (away_score > home_score) if pick == 'away' else (home_score > away_score)
        if away_score == home_score:
            return 0.0, 'push'
        profit = bet_size * (american_to_decimal(odds) - 1.0) if won else -bet_size
        return profit, ('win' if won else 'loss')

    elif market_type == 'SPREAD':
        line = float(row['Pinnacle_Away_Spread'] if pick == 'away' else row['Pinnacle_Home_Spread'])
        margin = (away_score - home_score) if pick == 'away' else (home_score - away_score)
        if margin + line == 0:
            return 0.0, 'push'
        won = (margin + line) > 0
        profit = bet_size * (100 / 110) if won else -bet_size  # Standard -110 juice
        return profit, ('win' if won else 'loss')

    elif market_type == 'TOTAL':
        total_line = float(row['Pinnacle_Total_Line'])
        if actual_total == total_line:
            return 0.0, 'push'
        won = (actual_total > total_line) if pick == 'over' else (actual_total < total_line)
        profit = bet_size * (100 / 110) if won else -bet_size
        return profit, ('win' if won else 'loss')

def run_backtest(target_date='2026-09-13', bet_size=25.0):
    print("Loading statistical baselines...")
    epa_stats = get_blended_nfl_stats(prior_season=2025, current_season=2026)

    games_evaluated = 0
    total_staked = 0.0
    total_profit = 0.0
    results_summary = {'ML': {'W': 0, 'L': 0, 'P': 0}, 'SPREAD': {'W': 0, 'L': 0, 'P': 0}, 'TOTAL': {'W': 0, 'L': 0, 'P': 0}}

    with open('history.csv', mode='r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))

    print(f"\n--- REPLAYING SLATE FOR {target_date} WITH NEW CALIBRATION ---\n")

    for row in reader:
        if row['Date'] != target_date or row.get('Actual_Away_Score') in ['N/A', '', None]:
            continue

        away = row['Away_Team']
        home = row['Home_Team']
        if away not in epa_stats or home not in epa_stats:
            continue

        games_evaluated += 1
        total_line = float(row['Pinnacle_Total_Line']) if row['Pinnacle_Total_Line'] != 'N/A' else None
        spread_line = float(row['Pinnacle_Home_Spread']) if row['Pinnacle_Home_Spread'] != 'N/A' else None

        # 1. Run simulation with new net EPA weighting & pace logic
        sim_res = simulate_nfl_game(epa_stats[away], epa_stats[home], total_line, spread_line)

        # 2. Apply the new 50/50 Bayesian shrinkage toward Pinnacle
        model_weight = 0.50
        away_ml = float(row['Pinnacle_Away_ML']) if row['Pinnacle_Away_ML'] != 'N/A' else None
        home_ml = float(row['Pinnacle_Home_ML']) if row['Pinnacle_Home_ML'] != 'N/A' else None

        if away_ml and home_ml:
            t_away, t_home = proportional_devig(away_ml, home_ml)
            sim_res["away_win_prob"] = (model_weight * (sim_res["away_win_prob"] / 100.0)) + ((1.0 - model_weight) * t_away)
            sim_res["home_win_prob"] = (model_weight * (sim_res["home_win_prob"] / 100.0)) + ((1.0 - model_weight) * t_home)

            away_ml_ev = calculate_ev(sim_res["away_win_prob"] * 100, away_ml)
            home_ml_ev = calculate_ev(sim_res["home_win_prob"] * 100, home_ml)

            # Check Moneyline +EV bet
            if away_ml_ev and away_ml_ev > 0:
                p, res = evaluate_bet('ML', 'away', row, bet_size)
                total_staked += bet_size
                total_profit += p
                results_summary['ML'][res[0].upper()] += 1
            elif home_ml_ev and home_ml_ev > 0:
                p, res = evaluate_bet('ML', 'home', row, bet_size)
                total_staked += bet_size
                total_profit += p
                results_summary['ML'][res[0].upper()] += 1

        # Check Spread +EV bet
        if spread_line is not None and row['Pinnacle_Away_Spread'] != 'N/A':
            t_sp_a, t_sp_h = proportional_devig(-110, -110)
            sim_res["spread_probs"]["away"] = (model_weight * sim_res["spread_probs"]["away"]) + ((1.0 - model_weight) * t_sp_a)
            sim_res["spread_probs"]["home"] = (model_weight * sim_res["spread_probs"]["home"]) + ((1.0 - model_weight) * t_sp_h)

            away_sp_ev = calculate_ev(sim_res["spread_probs"]["away"] * 100, -110, sim_res["spread_probs"]["push"] * 100)
            home_sp_ev = calculate_ev(sim_res["spread_probs"]["home"] * 100, -110, sim_res["spread_probs"]["push"] * 100)

            if away_sp_ev and away_sp_ev > 0:
                p, res = evaluate_bet('SPREAD', 'away', row, bet_size)
                total_staked += bet_size
                total_profit += p
                results_summary['SPREAD'][res[0].upper()] += 1
            elif home_sp_ev and home_sp_ev > 0:
                p, res = evaluate_bet('SPREAD', 'home', row, bet_size)
                total_staked += bet_size
                total_profit += p
                results_summary['SPREAD'][res[0].upper()] += 1

        # Check Totals +EV bet
        if total_line is not None:
            t_ou_o, t_ou_u = proportional_devig(-110, -110)
            sim_res["ou_probs"]["over"] = (model_weight * sim_res["ou_probs"]["over"]) + ((1.0 - model_weight) * t_ou_o)
            sim_res["ou_probs"]["under"] = (model_weight * sim_res["ou_probs"]["under"]) + ((1.0 - model_weight) * t_ou_u)

            over_ev = calculate_ev(sim_res["ou_probs"]["over"] * 100, -110, sim_res["ou_probs"]["push"] * 100)
            under_ev = calculate_ev(sim_res["ou_probs"]["under"] * 100, -110, sim_res["ou_probs"]["push"] * 100)

            if over_ev and over_ev > 0:
                p, res = evaluate_bet('TOTAL', 'over', row, bet_size)
                total_staked += bet_size
                total_profit += p
                results_summary['TOTAL'][res[0].upper()] += 1
            elif under_ev and under_ev > 0:
                p, res = evaluate_bet('TOTAL', 'under', row, bet_size)
                total_staked += bet_size
                total_profit += p
                results_summary['TOTAL'][res[0].upper()] += 1

    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    print(f"Games Evaluated: {games_evaluated}")
    print(f"Moneyline Record: {results_summary['ML']['W']}-{results_summary['ML']['L']}")
    print(f"Spread Record:    {results_summary['SPREAD']['W']}-{results_summary['SPREAD']['L']}")
    print(f"Totals Record:    {results_summary['TOTAL']['W']}-{results_summary['TOTAL']['L']}")
    print(f"Total Staked:     ${total_staked:.2f}")
    print(f"Net Profit:       ${total_profit:.2f}")
    print(f"Recalibrated ROI: {roi:.2f}%")

if __name__ == '__main__':
    run_backtest()
