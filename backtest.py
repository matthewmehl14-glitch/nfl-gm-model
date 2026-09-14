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

def evaluate_bet(market_type, pick, row, bet_amount):
    away_score = float(row['Actual_Away_Score'])
    home_score = float(row['Actual_Home_Score'])
    actual_total = away_score + home_score

    if market_type == 'ML':
        odds = float(row['Pinnacle_Away_ML'] if pick == 'away' else row['Pinnacle_Home_ML'])
        won = (away_score > home_score) if pick == 'away' else (home_score > away_score)
        if away_score == home_score:
            return 0.0, 'push'
        profit = bet_amount * (american_to_decimal(odds) - 1.0) if won else -bet_amount
        return profit, ('win' if won else 'loss')

    elif market_type == 'SPREAD':
        line = float(row['Pinnacle_Away_Spread'] if pick == 'away' else row['Pinnacle_Home_Spread'])
        margin = (away_score - home_score) if pick == 'away' else (home_score - away_score)
        if margin + line == 0:
            return 0.0, 'push'
        won = (margin + line) > 0
        profit = bet_amount * (100 / 110) if won else -bet_amount  # Standard -110 juice
        return profit, ('win' if won else 'loss')

    elif market_type == 'TOTAL':
        total_line = float(row['Pinnacle_Total_Line'])
        if actual_total == total_line:
            return 0.0, 'push'
        won = (actual_total > total_line) if pick == 'over' else (actual_total < total_line)
        profit = bet_amount * (100 / 110) if won else -bet_amount
        return profit, ('win' if won else 'loss')

def run_backtest(target_date='2026-09-13', base_unit_dollars=25.0):
    # Adjusting parameters for early-season variance
    model_weight = 0.15 
    min_ev_hurdle = 3.0
    max_ev_hurdle = 10.0
    
    print("Loading statistical baselines...")
    epa_stats = get_blended_nfl_stats(prior_season=2025, current_season=2026)

    games_evaluated = 0
    total_staked = 0.0
    total_profit = 0.0
    results_summary = {'ML': {'W': 0, 'L': 0, 'P': 0}, 'SPREAD': {'W': 0, 'L': 0, 'P': 0}, 'TOTAL': {'W': 0, 'L': 0, 'P': 0}}

    with open('history.csv', mode='r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))

    print(f"\n--- REPLAYING SLATE FOR {target_date} WITH KELLY STAKING & EV CAPS ---\n")

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

        sim_res = simulate_nfl_game(epa_stats[away], epa_stats[home], total_line, spread_line)

        away_ml = float(row['Pinnacle_Away_ML']) if row['Pinnacle_Away_ML'] != 'N/A' else None
        home_ml = float(row['Pinnacle_Home_ML']) if row['Pinnacle_Home_ML'] != 'N/A' else None

        if away_ml and home_ml:
            t_away, t_home = proportional_devig(away_ml, home_ml)
            sim_res["away_win_prob"] = (model_weight * (sim_res["away_win_prob"] / 100.0)) + ((1.0 - model_weight) * t_away)
            sim_res["home_win_prob"] = (model_weight * (sim_res["home_win_prob"] / 100.0)) + ((1.0 - model_weight) * t_home)

            away_ml_ev = calculate_ev(sim_res["away_win_prob"] * 100, away_ml)
            home_ml_ev = calculate_ev(sim_res["home_win_prob"] * 100, home_ml)

            if away_ml_ev and min_ev_hurdle <= away_ml_ev <= max_ev_hurdle:
                units = calc_kelly_units(sim_res["away_win_prob"] * 100, away_ml)
                bet_amt = units * base_unit_dollars
                if bet_amt > 0:
                    p, res = evaluate_bet('ML', 'away', row, bet_amt)
                    total_staked += bet_amt
                    total_profit += p
                    results_summary['ML'][res[0].upper()] += 1
                    print(f"Betted ML: {away} ({res.upper()}) | EV: {away_ml_ev}% | Stake: ${bet_amt:.2f} ({units}u) | Profit: ${p:.2f}")

            elif home_ml_ev and min_ev_hurdle <= home_ml_ev <= max_ev_hurdle:
                units = calc_kelly_units(sim_res["home_win_prob"] * 100, home_ml)
                bet_amt = units * base_unit_dollars
                if bet_amt > 0:
                    p, res = evaluate_bet('ML', 'home', row, bet_amt)
                    total_staked += bet_amt
                    total_profit += p
                    results_summary['ML'][res[0].upper()] += 1
                    print(f"Betted ML: {home} ({res.upper()}) | EV: {home_ml_ev}% | Stake: ${bet_amt:.2f} ({units}u) | Profit: ${p:.2f}")

        if spread_line is not None and row['Pinnacle_Away_Spread'] != 'N/A':
            t_sp_a, t_sp_h = proportional_devig(-110, -110)
            sim_res["spread_probs"]["away"] = (model_weight * sim_res["spread_probs"]["away"]) + ((1.0 - model_weight) * t_sp_a)
            sim_res["spread_probs"]["home"] = (model_weight * sim_res["spread_probs"]["home"]) + ((1.0 - model_weight) * t_sp_h)

            away_sp_ev = calculate_ev(sim_res["spread_probs"]["away"] * 100, -110, sim_res["spread_probs"]["push"] * 100)
            home_sp_ev = calculate_ev(sim_res["spread_probs"]["home"] * 100, -110, sim_res["spread_probs"]["push"] * 100)

            if away_sp_ev and min_ev_hurdle <= away_sp_ev <= max_ev_hurdle:
                units = calc_kelly_units(sim_res["spread_probs"]["away"] * 100, -110, sim_res["spread_probs"]["push"] * 100)
                bet_amt = units * base_unit_dollars
                if bet_amt > 0:
                    p, res = evaluate_bet('SPREAD', 'away', row, bet_amt)
                    total_staked += bet_amt
                    total_profit += p
                    results_summary['SPREAD'][res[0].upper()] += 1
                    print(f"Betted SPREAD: {away} ({res.upper()}) | EV: {away_sp_ev}% | Stake: ${bet_amt:.2f} ({units}u) | Profit: ${p:.2f}")

            elif home_sp_ev and min_ev_hurdle <= home_sp_ev <= max_ev_hurdle:
                units = calc_kelly_units(sim_res["spread_probs"]["home"] * 100, -110, sim_res["spread_probs"]["push"] * 100)
                bet_amt = units * base_unit_dollars
                if bet_amt > 0:
                    p, res = evaluate_bet('SPREAD', 'home', row, bet_amt)
                    total_staked += bet_amt
                    total_profit += p
                    results_summary['SPREAD'][res[0].upper()] += 1
                    print(f"Betted SPREAD: {home} ({res.upper()}) | EV: {home_sp_ev}% | Stake: ${bet_amt:.2f} ({units}u) | Profit: ${p:.2f}")

        if total_line is not None:
            t_ou_o, t_ou_u = proportional_devig(-110, -110)
            sim_res["ou_probs"]["over"] = (model_weight * sim_res["ou_probs"]["over"]) + ((1.0 - model_weight) * t_ou_o)
            sim_res["ou_probs"]["under"] = (model_weight * sim_res["ou_probs"]["under"]) + ((1.0 - model_weight) * t_ou_u)

            over_ev = calculate_ev(sim_res["ou_probs"]["over"] * 100, -110, sim_res["ou_probs"]["push"] * 100)
            under_ev = calculate_ev(sim_res["ou_probs"]["under"] * 100, -110, sim_res["ou_probs"]["push"] * 100)

            if over_ev and min_ev_hurdle <= over_ev <= max_ev_hurdle:
                units = calc_kelly_units(sim_res["ou_probs"]["over"] * 100, -110, sim_res["ou_probs"]["push"] * 100)
                bet_amt = units * base_unit_dollars
                if bet_amt > 0:
                    p, res = evaluate_bet('TOTAL', 'over', row, bet_amt)
                    total_staked += bet_amt
                    total_profit += p
                    results_summary['TOTAL'][res[0].upper()] += 1
                    print(f"Betted TOTAL: OVER {total_line} in {away}@{home} ({res.upper()}) | EV: {over_ev}% | Stake: ${bet_amt:.2f} ({units}u) | Profit: ${p:.2f}")

            elif under_ev and min_ev_hurdle <= under_ev <= max_ev_hurdle:
                units = calc_kelly_units(sim_res["ou_probs"]["under"] * 100, -110, sim_res["ou_probs"]["push"] * 100)
                bet_amt = units * base_unit_dollars
                if bet_amt > 0:
                    p, res = evaluate_bet('TOTAL', 'under', row, bet_amt)
                    total_staked += bet_amt
                    total_profit += p
                    results_summary['TOTAL'][res[0].upper()] += 1
                    print(f"Betted TOTAL: UNDER {total_line} in {away}@{home} ({res.upper()}) | EV: {under_ev}% | Stake: ${bet_amt:.2f} ({units}u) | Profit: ${p:.2f}")

    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    
    print("\n--- FINAL BACKTEST RESULTS ---")
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
