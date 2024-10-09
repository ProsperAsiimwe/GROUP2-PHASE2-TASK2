import argparse
import time
import art
import os
import numpy as np
import pandas as pd

# Import functions from logic.py
from logic import generate_learned_cpts, investment_recommendation, value_network, quality_network
from invest.preprocessing.dataloader import load_data

VERSION = 1.0

def main_with_learned_cpts(algorithm='MLE'):
    print(f"[INFO] Generating learned CPTs using {algorithm}.")
    # Load data and generate CPTs
    df_ = load_data()
    learned_cpts = generate_learned_cpts(df_, algorithm=algorithm)
    main(learned_cpts=learned_cpts)

def main(learned_cpts=None):
    # Track experiment runtime
    start = time.time()

    # Load the dataset
    df_ = load_data()
    if df_.empty:
        print("[ERROR] No data loaded. Exiting.")
        return

    print(f"[INFO] Using learned CPTs: {learned_cpts is not None}")

    # Iterate through each company in the dataset and apply the decision-making logic
    for year in range(args.start, args.end + 1):
        print(f"\n[INFO] Processing year: {year}")
        for company in df_['Company'].unique():
            print(f"[INFO] Evaluating company: {company} for year {year}.")

            # Extract relevant states for Value and Quality Networks
            company_data = df_[(df_['Company'] == company) & (df_['Year'] == year)]
            if company_data.empty:
                print(f"[WARNING] No data available for {company} in year {year}. Skipping.")
                continue

            # Extract states for the Value Network (assuming column names match node names)
            pe_relative_market_state = company_data['PERelative_ShareMarket'].values[0]
            pe_relative_sector_state = company_data['PERelative_ShareSector'].values[0]
            forward_pe_state = company_data['ForwardPE_CurrentVsHistory'].values[0]
            future_performance_state = company_data['FutureSharePerformance'].values[0]

            # Run Value Network
            value_decision = value_network(pe_relative_market_state, pe_relative_sector_state, forward_pe_state,
                                           future_performance_state=future_performance_state, learned_cpts=learned_cpts)

            # Extract states for the Quality Network
            roe_vs_coe_state = company_data['ROE_vs_COE'].values[0]
            relative_debt_equity_state = company_data['Relative_Debt_Equity'].values[0]
            cagr_vs_inflation_state = company_data['CAGR_vs_Inflation'].values[0]
            systematic_risk_state = company_data['Systematic_Risk'].values[0]

            # Run Quality Network
            quality_decision = quality_network(roe_vs_coe_state, relative_debt_equity_state, cagr_vs_inflation_state,
                                               systematic_risk_state=systematic_risk_state, learned_cpts=learned_cpts)

            # Run Investment Recommendation Network
            investment_decision = investment_recommendation(value_decision, quality_decision, learned_cpts=learned_cpts)

            print(f"[INFO] Investment decision for {company} in {year}: {investment_decision}")

    # Calculate experiment runtime
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nExperiment Time: ""{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # Set up argument parser for command-line options
    parser = argparse.ArgumentParser(description='Intelligent system for automated share evaluation', epilog='Version 1.0')
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=2018)
    parser.add_argument("--margin_of_safety", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=1.6)
    parser.add_argument("--extension", type=str2bool, default=False)
    parser.add_argument("--noise", type=str2bool, default=False)
    parser.add_argument("--ablation", type=str2bool, default=False)
    parser.add_argument("--network", type=str, default='v')
    parser.add_argument("--gnn", type=str2bool, default=False)
    parser.add_argument("--holding_period", type=int, default=-1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--cpt_algorithm", type=str, choices=['MLE', 'EM', 'MDL'], default='MLE',
                        help="Algorithm to use for learning CPTs.")
    args = parser.parse_args()

    # Display the banner
    print(art.text2art("INVEST"))
    print("Insaaf Dhansay & Kialan Pillay")
    print("© University of Cape Town 2021")
    print("Version {}".format(VERSION))
    print("=" * 50)

    # Handle experiment execution
    if args.noise:
        print("[INFO] Running experiments with noise...")
    else:
        print(f"[INFO] Running main with CPT learning algorithm: {args.cpt_algorithm}.")
        main_with_learned_cpts(algorithm=args.cpt_algorithm)
