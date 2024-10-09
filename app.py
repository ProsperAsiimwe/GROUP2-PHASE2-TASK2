import argparse
import time
import art
import os
import numpy as np
import pyAgrum as gum
import pandas as pd

from invest.decision import investment_portfolio
from invest.preprocessing.clean import merge, clean  # Importing from the preprocessing scripts
from invest.preprocessing.simulation import simulate  # Importing the simulate function
from invest.preprocessing.dataloader import load_data


VERSION = 1.0


def assign_labels(variable, labels):
    """
    Helper function to assign a list of labels to a LabelizedVariable in pyAgrum.
    """
    for index, label in enumerate(labels):
        variable.changeLabel(index, label)
    return variable


def initialize_baseline_bn():
    """
    Initializes the Bayesian Network with refined baseline CPTs based on the investment recommendation structure.
    """
    bn = gum.BayesNet("InvestNetwork")

    # Define the node states based on the experiment designs
    nodes = {
        'ValueRelativeToPrice': ['Cheap', 'FairValue', 'Expensive'],
        'Expensive_E': ['No', 'Yes'],
        'FutureSharePerformance': ['Positive', 'Stagnant', 'Negative'],
        'PERelative_ShareMarket': ['Cheap', 'FairValue', 'Expensive'],
        'PERelative_ShareSector': ['Cheap', 'FairValue', 'Expensive']
    }

    print(f"[DEBUG] Creating nodes and states: {nodes}")
    for node_name, states in nodes.items():
        variable = gum.LabelizedVariable(node_name, '', len(states))
        for idx, state in enumerate(states):
            variable.changeLabel(idx, state)
        bn.add(variable)
        print(f"[DEBUG] Added node '{node_name}' with states: {states}")

    # Add arcs to establish parent-child relationships
    bn.addArc('FutureSharePerformance', 'Expensive_E')
    bn.addArc('FutureSharePerformance', 'ValueRelativeToPrice')
    bn.addArc('PERelative_ShareMarket', 'ValueRelativeToPrice')
    bn.addArc('PERelative_ShareSector', 'ValueRelativeToPrice')

    print(f"[DEBUG] Added arcs to define parent-child relationships in the network.")

    # Initialize baseline CPTs with structured probability distributions
    baseline_cpts = {
        'FutureSharePerformance': [0.4, 0.3, 0.3],
        'PERelative_ShareMarket': [0.35, 0.45, 0.2],
        'PERelative_ShareSector': [0.25, 0.5, 0.25],
        'Expensive_E': [[0.1, 0.9], [0.4, 0.6], [0.7, 0.3]]  # Example probabilities for 'Expensive_E'
    }

    # Refine CPT for ValueRelativeToPrice considering parent nodes and experimental insights
    value_cpt_shape = (3, 3, 3, 3)
    value_cpt = np.zeros(value_cpt_shape)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Structure the probabilities based on specific parent states
                if i == 0 and j == 0 and k == 0:
                    value_cpt[i, j, k, :] = [0.75, 0.2, 0.05]  # Predominantly Cheap
                elif i == 2 and j == 2 and k == 2:
                    value_cpt[i, j, k, :] = [0.1, 0.15, 0.75]  # Predominantly Expensive
                else:
                    value_cpt[i, j, k, :] = [0.33, 0.34, 0.33]  # Default to neutral distribution

    print(f"[DEBUG] Full CPT matrix for ValueRelativeToPrice:\n{value_cpt}")
    baseline_cpts['ValueRelativeToPrice'] = value_cpt

    print(f"[DEBUG] Applying baseline CPTs to the network...")
    bn.cpt('FutureSharePerformance').fillWith(baseline_cpts['FutureSharePerformance'])
    bn.cpt('PERelative_ShareMarket').fillWith(baseline_cpts['PERelative_ShareMarket'])
    bn.cpt('PERelative_ShareSector').fillWith(baseline_cpts['PERelative_ShareSector'])

    try:
        bn.cpt('ValueRelativeToPrice').fillWith(value_cpt.flatten())
        print(f"[INFO] ValueRelativeToPrice CPT filled successfully.")
    except Exception as e:
        print(f"[ERROR] Error while filling ValueRelativeToPrice CPT: {e}")

    return bn, baseline_cpts


def walk_forward_cpt_learning(df, bn, baseline_cpts, algorithm='MLE'):
    """
    Applies walk-forward CPT learning using different algorithms and compares with baseline CPTs.
    """
    learned_cpts = {}
    learner = gum.BNLearner(df, bn)

    # Adjust smoothing parameter based on experiment feedback
    learner.useSmoothingPrior(2.0)  # Reduce smoothing to avoid over-smoothed probabilities

    # Apply different algorithms for parameter learning
    if algorithm == 'MLE':
        print("[INFO] Learning CPTs using Maximum Likelihood Estimation (MLE).")
        learner.learnParameters(bn.dag())
    elif algorithm == 'EM':
        print("[INFO] Learning CPTs using Expectation-Maximization (EM).")
        learner.useEM(epsilon=1e-5)  # Adjust EM convergence threshold
        learner.learnParameters(bn.dag())
    elif algorithm == 'MDL':
        print("[INFO] Learning CPTs using Minimum Description Length (MDL).")
        learner.useMDLCorrection()
        learner.learnParameters(bn.dag())
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Please use 'MLE', 'EM', or 'MDL'.")

    # Check for missing parent combinations and zero frequencies
    debug_zero_frequencies(df, bn)

    # Extract learned CPTs and compare with baseline
    for node in bn.names():
        learned_cpt = bn.cpt(node).tolist()
        baseline_cpt = baseline_cpts.get(node)
        learned_cpts[node] = learned_cpt
        print(f"[INFO] Node '{node}' - Learned CPT: {learned_cpt}, Baseline CPT: {baseline_cpt}")

    return learned_cpts


def debug_zero_frequencies(df, bn):
    """
    Debug function to identify zero frequencies in parent-child combinations.
    """
    for node in bn.names():
        parent_indices = bn.parents(node)  # Retrieve parent indices
        parents = [bn.variable(p).name() for p in parent_indices]  # Correctly get the variable names

        print(f"[DEBUG] Checking frequencies for node '{node}' with parents: {parents}")

        if not parents:
            print(f"[DEBUG] Node '{node}' has no parents, skipping zero frequency check.")
            continue

        # Use the correct parent names for indexing in the DataFrame
        parent_combinations = df[parents].drop_duplicates()

        for _, combination in parent_combinations.iterrows():
            count = len(df[(df[parents] == combination).all(axis=1)])
            print(f"[DEBUG] Combination {dict(combination)} appears {count} times in the dataset.")




def derive_missing_variables(df):
    """
    Derive the missing variables required by the Bayesian Network using existing columns.
    """
    # Rule for ValueRelativeToPrice based on PE
    df['ValueRelativeToPrice'] = df['PE'].apply(lambda x: 'Cheap' if x < 10 else ('FairValue' if 10 <= x < 20 else 'Expensive'))

    # Rule for FutureSharePerformance based on percentage change in Price
    df['FutureSharePerformance'] = df['Price'].pct_change().apply(
        lambda x: 'Positive' if x > 0 else ('Stagnant' if x == 0 else 'Negative')
    )

    # Fill the first row NaN value for FutureSharePerformance
    df['FutureSharePerformance'].iloc[0] = 'Stagnant'  # Default to 'Stagnant' for the first row

    # Rule for Expensive_E based on ValueRelativeToPrice
    df['Expensive_E'] = df['ValueRelativeToPrice'].apply(lambda x: 'Yes' if x == 'Expensive' else 'No')

    # Rule for PERelative_ShareMarket based on PE and PEMarket
    df['PERelative_ShareMarket'] = df.apply(lambda row: 'Cheap' if row['PE'] < row['PEMarket'] else (
        'FairValue' if row['PE'] == row['PEMarket'] else 'Expensive'), axis=1)

    # Rule for PERelative_ShareSector based on PE and PESector
    df['PERelative_ShareSector'] = df.apply(lambda row: 'Cheap' if row['PE'] < row['PESector'] else (
        'FairValue' if row['PE'] == row['PESector'] else 'Expensive'), axis=1)

    return df

def main_with_learned_cpts(algorithm='MLE'):
    print(f"[INFO] Generating learned CPTs using {algorithm}.")

    # Step 1: Preprocess and clean the data
    print("[INFO] Preprocessing and cleaning the dataset...")
    merge(args)  # Merging and cleaning raw data files
    df = pd.read_csv('data/INVEST_clean.csv')

    # Step 2: Optionally, apply noise to the dataset if specified
    if args.noise:
        print("[INFO] Adding noise to the dataset...")
        df = simulate(df, frac=0.3, scale=1, method='std')

    # Step 3: Derive missing columns for the Bayesian Network
    df = derive_missing_variables(df)
    print(f"[INFO] Derived missing variables in the dataset: {df[['ValueRelativeToPrice', 'FutureSharePerformance', 'Expensive_E', 'PERelative_ShareMarket', 'PERelative_ShareSector']].head()}")

    # Verify if the columns are added successfully
    print(f"[DEBUG] Dataset columns after deriving variables: {df.columns.tolist()}")

    # Step 4: Initialize baseline network and learn CPTs
    bn, baseline_cpts = initialize_baseline_bn()
    learned_cpts = walk_forward_cpt_learning(df, bn, baseline_cpts, algorithm=algorithm)

    print("[INFO] Learned CPTs generated successfully.")
    return learned_cpts

def main(learned_cpts=None):
    start = time.time()
    df_ = load_data()
    
    if df_.empty:
        print("[ERROR] No data loaded. Exiting.")
        return

    print(f"[INFO] Using learned CPTs: {learned_cpts is not None}")

    # Pass learned CPTs into the investment_portfolio function
    jgind_portfolio = investment_portfolio(df_, args, "JGIND", True, learned_cpts=learned_cpts)
    jcsev_portfolio = investment_portfolio(df_, args, "JCSEV", True, learned_cpts=learned_cpts)

    end = time.time()
    jgind_metrics_ = list(jgind_portfolio["ip"].values())[2::]
    jcsev_metrics_ = list(jcsev_portfolio["ip"].values())[2::]

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExperiment Time: ""{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    return jgind_metrics_, jcsev_metrics_

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
    parser = argparse.ArgumentParser(description='Intelligent system for automated share evaluation',
                                     epilog='Version 1.0')
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
    parser.add_argument("--raw_folder", type=str, default='data/INVEST_IRESS')
    parser.add_argument("--output", type=str, default='data/INVEST')

    args = parser.parse_args()

    print(art.text2art("INVEST"))
    print("Insaaf Dhansay & Kialan Pillay")
    print("© University of Cape Town 2021")
    print("Version {}".format(VERSION))
    print("=" * 50)

    if args.noise:
        print("[INFO] Running experiments with noise...")
        jgind_metrics = []
        jcsev_metrics = []
        for i in range(0, 10):
            ratios_jgind, ratios_jcsev = main()
            jgind_metrics.append(ratios_jgind)
            jcsev_metrics.append(ratios_jcsev)
        jgind_averaged_metrics = np.mean(jgind_metrics, axis=0)
        jcsev_averaged_metrics = np.mean(jcsev_metrics, axis=0)

        for i in range(0, 2):
            jgind_averaged_metrics[i] *= 100
            jcsev_averaged_metrics[i] *= 100
        print("JGIND", [round(v, 2) for v in jgind_averaged_metrics])
        print("JCSEV", [round(v, 2) for v in jcsev_averaged_metrics])
    else:
        print(f"[INFO] Running main with CPT learning algorithm: {args.cpt_algorithm}.")
        main_with_learned_cpts(algorithm=args.cpt_algorithm)