import json
import pandas as pd
import invest.evaluation.validation as validation
from invest.networks.invest_recommendation import investment_recommendation
from invest.networks.quality_evaluation import quality_network
from invest.networks.value_evaluation import value_network
from invest.preprocessing.simulation import simulate
from invest.store import Store

# Load sector companies from JSON files
companies_jcsev = json.load(open('data/jcsev.json'))['names']
companies_jgind = json.load(open('data/jgind.json'))['names']
companies = companies_jcsev + companies_jgind
companies_dict = {"JCSEV": companies_jcsev, "JGIND": companies_jgind}

def investment_portfolio(df_, params, index_code, verbose=False, learned_cpts=None):
    """
    Decides the shares for inclusion in an investment portfolio using INVEST Bayesian networks.
    Computes performance metrics for the investment portfolio and benchmark index.

    Parameters
    ----------
    df_ : pandas.DataFrame
        Fundamental and price data.
    params : argparse.Namespace
        Command line arguments.
    index_code: str
        Johannesburg Stock Exchange sector index code (e.g., 'JCSEV' or 'JGIND').
    verbose: bool, optional
        Print output to console.
    learned_cpts: dict, optional
        Dictionary of learned CPTs to be used in the Bayesian networks.

    Returns
    -------
    portfolio: dict
        A dictionary containing investment portfolio and benchmark metrics.
    """
    print(f"[INFO] Constructing the investment portfolio for {index_code} from {params.start} to {params.end}.")
    
    # Use simulation if noise parameter is set; otherwise, use original data
    if params.noise:
        df = simulate(df_)
        print(f"[INFO] Data simulation applied due to noise setting.")
    else:
        df = df_

    # Initialize portfolio data structures
    prices_initial = {}
    prices_current = {}
    betas = {}
    investable_shares = {}

    # Iterate through each year in the given period
    for year in range(params.start, params.end):
        print(f"[INFO] Processing year: {year}")
        
        store = Store(df, companies, companies_jcsev, companies_jgind, params.margin_of_safety, params.beta, year, False)
        investable_shares[str(year)] = []
        prices_initial[str(year)] = []
        prices_current[str(year)] = []
        betas[str(year)] = []

        # Iterate through companies in the specified index
        for company in companies_dict[index_code]:
            if store.get_acceptable_stock(company):
                print(f"[INFO] Evaluating company: {company} for year {year}.")

                # Call investment decision with learned CPTs if provided
                decision = investment_decision(store, company, params.extension, params.ablation, params.network, learned_cpts)
                print(f"[INFO] Investment decision for {company} in {year}: {decision}")

                if decision == "Yes":
                    # Filter data for the specific year and company
                    mask = (df_['Date'] >= str(year) + '-01-01') & (df_['Date'] <= str(year) + '-12-31') & (df_['Name'] == company)
                    df_year = df_[mask]

                    # Append investable shares and prices
                    investable_shares[str(year)].append(company)
                    prices_initial[str(year)].append(df_year.iloc[0]['Price'])
                    prices_current[str(year)].append(df_year.iloc[params.holding_period]['Price'])
                    betas[str(year)].append(df_year.iloc[params.holding_period]["ShareBeta"])

    # Verbose output for debugging and analysis
    if verbose:
        print(f"\n{index_code} Portfolio from {params.start} to {params.end}")
        print("-" * 50)
        print(f"\nInvestable Shares by Year: {investable_shares}")

    # Calculate performance metrics for the investment portfolio
    ip_ar, ip_cr, ip_aar, ip_treynor, ip_sharpe = validation.process_metrics(
        df, prices_initial, prices_current, betas, params.start, params.end, index_code
    )
    benchmark_ar, benchmark_cr, benchmark_aar, benchmark_treynor, benchmark_sharpe = \
        validation.process_benchmark_metrics(params.start, params.end, index_code, params.holding_period)

    print(f"[INFO] Investment portfolio metrics calculated successfully.")

    # Return the final portfolio with both investment and benchmark metrics
    portfolio = {
        "ip": {
            "shares": investable_shares,
            "annualReturns": ip_ar,
            "compoundReturn": ip_cr,
            "averageAnnualReturn": ip_aar,
            "treynor": ip_treynor,
            "sharpe": ip_sharpe,
        },
        "benchmark": {
            "annualReturns": benchmark_ar,
            "compoundReturn": benchmark_cr,
            "averageAnnualReturn": benchmark_aar,
            "treynor": benchmark_treynor,
            "sharpe": benchmark_sharpe,
        }
    }

    return portfolio

def investment_decision(store, company, extension=False, ablation=False, network='v', learned_cpts=None):
    """
    Returns an investment decision for shares of the specified company.

    Parameters
    ----------
    store : Store
        Ratio and threshold data store.
    company : str
        Company to evaluate.
    extension: bool, optional
        Use Quality Network systematic risk extension.
    ablation: bool, optional
        Conduct ablation test.
    network: str, optional
        'v' for Value network, 'q' for Quality network.
    learned_cpts: dict, optional
        Dictionary of learned CPTs to be used in the networks.

    Returns
    -------
    str : Investment decision (Yes or No).
    """
    print(f"[INFO] Running investment decision for {company}.")

    # Retrieve metrics from the store
    pe_relative_market = store.get_pe_relative_market(company)
    pe_relative_sector = store.get_pe_relative_sector(company)
    forward_pe = store.get_forward_pe(company)
    roe_vs_coe = store.get_roe_vs_coe(company)
    relative_debt_equity = store.get_relative_debt_equity(company)
    cagr_vs_inflation = store.get_cagr_vs_inflation(company)
    systematic_risk = store.get_systematic_risk(company)

    # Evaluate using the value and quality networks
    value_decision = value_network(pe_relative_market, pe_relative_sector, forward_pe, None, learned_cpts)
    quality_decision = quality_network(roe_vs_coe, relative_debt_equity, cagr_vs_inflation, systematic_risk, extension, learned_cpts)
    print(f"[INFO] Value Decision: {value_decision}, Quality Decision: {quality_decision}")

    # Get the final investment recommendation based on value and quality decisions
    return investment_recommendation(value_decision, quality_decision, learned_cpts)
