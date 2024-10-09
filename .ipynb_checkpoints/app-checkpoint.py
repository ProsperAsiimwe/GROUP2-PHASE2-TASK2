import argparse
import time
import art
import os
import numpy as np
import pyAgrum as gum
import pandas as pd

from invest.decision import investment_portfolio
from invest.preprocessing.dataloader import load_data

VERSION = 1.0

# Function to generate learned CPTs using different algorithms
def generate_learned_cpts(df, algorithm='MLE'):
    """
    Generate learned CPTs for a given dataset using a specified learning algorithm.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to learn from.
    algorithm : str
        The algorithm to use for learning CPTs ('MLE', 'EM', 'MDL').

    Returns
    -------
    learned_cpts : dict
        Dictionary of learned CPTs for the nodes in the Bayesian Network.
    """
    learned_cpts = {}
    
    # Define your network structure here (adjust nodes as per your network)
    bn = gum.BayesNet("InvestNetwork")
    bn.add(gum.LabelizedVariable('Value', '', 3))  # Example node
    bn.add(gum.LabelizedVariable('Quality', '', 3))  # Example node
    bn.add(gum.LabelizedVariable('Performance', '', 3))  # Example node

    # Add arcs based on your model structure
    bn.addArc('Performance', 'Value')
    bn.addArc('Performance', 'Quality')

    # Fit the network using different algorithms
    if algorithm == 'MLE':
        print("[INFO] Learning CPTs using Maximum Likelihood Estimation (MLE).")
        learner = gum.BNLearner(df, bn)
        learner.useSmoothingPrior(1.0)
        learned_bn = learner.learnParameters(bn.dag())
    elif algorithm == 'EM':
        print("[INFO] Learning CPTs using Expectation-Maximization (EM).")
        em = gum.BNLearner(df, bn)
        em.useSmoothingPrior(1.0)
        learned_bn = em.learnParameters(bn.dag(), algo="em")
    elif algorithm == 'MDL':
        print("[INFO] Learning CPTs using Minimum Description Length (MDL).")
        mdl_learner = gum.BNLearner(df, bn)
        mdl_learner.useMDLCriterion()
        learned_bn = mdl_learner.learnParameters(bn.dag())
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Please use 'MLE', 'EM', or 'MDL'.")

    # Extract learned CPTs for relevant nodes
    for node in bn.names():
        learned_cpts[node] = learned_bn.cpt(node).tolist()

    print(f"[INFO] Learned CPTs for {algorithm} generated successfully.")
    return learned_cpts

def main(learned_cpts=None):
    start = time.time()
    print("[INFO] Loading data...")
    df_ = load_data()
    if df_.empty:
        print("[ERROR] No data loaded. Exiting.")
        return

    print(f"[INFO] Using learned CPTs: {learned_cpts is not None}")
    print("[INFO] Constructing JGIND portfolio...")
    jgind_portfolio = investment_portfolio(df_, args, "JGIND", True, learned_cpts=learned_cpts)
    print("[INFO] JGIND Portfolio constructed.")

    print("[INFO] Constructing JCSEV portfolio...")
    jcsev_portfolio = investment_portfolio(df_, args, "JCSEV", True, learned_cpts=learned_cpts)
    print("[INFO] JCSEV Portfolio constructed.")

    end = time.time()
    
    jgind_metrics_ = list(jgind_portfolio["ip"].values())[2::]
    jcsev_metrics_ = list(jcsev_portfolio["ip"].values())[2::]

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nExperiment Time: ""{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
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
    args = parser.parse_args()

    print(art.text2art("INVEST"))
    print("Insaaf Dhansay & Kialan Pillay")
    print("© University of Cape Town 2021")
    print("Version {}".format(VERSION))
    print("=" * 50)

    # Generate learned CPTs using different algorithms
    print("[INFO] Generating learned CPTs using MLE, EM, and MDL.")
    learned_cpts_mle = generate_learned_cpts(load_data(), algorithm='MLE')
    learned_cpts_em = generate_learned_cpts(load_data(), algorithm='EM')
    learned_cpts_mdl = generate_learned_cpts(load_data(), algorithm='MDL')

    # Run simulations for each set of learned CPTs
    print("[INFO] Running simulations for each learned CPT configuration.")
    print("\n[INFO] MLE Results:")
    main(learned_cpts=learned_cpts_mle)

    print("\n[INFO] EM Results:")
    main(learned_cpts=learned_cpts_em)

    print("\n[INFO] MDL Results:")
    main(learned_cpts=learned_cpts_mdl)
