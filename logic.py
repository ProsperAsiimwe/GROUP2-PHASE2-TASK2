import pyAgrum as gum
import numpy as np

def assign_labels(variable, labels):
    """
    Helper function to assign a list of labels to a LabelizedVariable in pyAgrum.

    Parameters
    ----------
    variable : gum.LabelizedVariable
        The variable to assign labels to.
    labels : list
        List of labels to assign to the variable.

    Returns
    -------
    gum.LabelizedVariable
        The variable with updated labels.
    """
    for index, label in enumerate(labels):
        variable.changeLabel(index, label)
    return variable

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

    # Define the Bayesian Network structure using actual node names
    bn = gum.BayesNet("InvestNetwork")

    # Create variables using the exact node names from the `value_network`
    value_var = assign_labels(gum.LabelizedVariable('ValueRelativeToPrice', '', 3), ["Cheap", "FairValue", "Expensive"])
    quality_var = assign_labels(gum.LabelizedVariable('Expensive_E', '', 2), ["No", "Yes"])
    performance_var = assign_labels(gum.LabelizedVariable('FutureSharePerformance', '', 3), ["Positive", "Stagnant", "Negative"])

    # Add variables to the Bayesian Network
    bn.add(value_var)
    bn.add(quality_var)
    bn.add(performance_var)

    # Add arcs based on your model structure
    bn.addArc('FutureSharePerformance', 'ValueRelativeToPrice')
    bn.addArc('FutureSharePerformance', 'Expensive_E')

    # Ensure the DataFrame has the necessary columns and correct labels
    required_columns = ['ValueRelativeToPrice', 'Expensive_E', 'FutureSharePerformance']
    if not set(required_columns).issubset(df.columns):
        print(f"[WARNING] DataFrame missing required columns for Bayesian Network: {required_columns}")
        print("[INFO] Creating necessary columns based on available data...")
        df['FutureSharePerformance'] = df.apply(lambda row: 'Positive' if row['Price'] > row['Open'] else 'Negative', axis=1)
        df['ValueRelativeToPrice'] = df['FutureSharePerformance'].apply(lambda x: 'FairValue' if x == 'Positive' else 'Cheap')
        df['Expensive_E'] = df['FutureSharePerformance'].apply(lambda x: 'No' if x == 'Positive' else 'Yes')

    # Map DataFrame values to string labels expected by the Bayesian Network
    label_mapping = {
        'ValueRelativeToPrice': {'FairValue': 'FairValue', 'Cheap': 'Cheap', 'Expensive': 'Expensive'},
        'Expensive_E': {'No': 'No', 'Yes': 'Yes'},
        'FutureSharePerformance': {'Positive': 'Positive', 'Stagnant': 'Stagnant', 'Negative': 'Negative'}
    }

    for col in required_columns:
        if col in df.columns:
            df[col] = df[col].map(label_mapping[col])

    print(f"[INFO] Mapped DataFrame values to correct string labels.")
    print(f"[INFO] Updated DataFrame for Learning:\n{df.head()}")

    # Fit the network using different algorithms
    learner = gum.BNLearner(df[required_columns], bn)
    learner.useSmoothingPrior(1.0)  # Apply smoothing to handle missing data

    if algorithm == 'MLE':
        print("[INFO] Learning CPTs using Maximum Likelihood Estimation (MLE).")
        learned_bn = learner.learnParameters(bn.dag())
    elif algorithm == 'EM':
        print("[INFO] Learning CPTs using Expectation-Maximization (EM).")
        learner.useEM(epsilon=1e-4)   # Activate EM learning
        learned_bn = learner.learnParameters(bn.dag())
    elif algorithm == 'MDL':
        print("[INFO] Learning CPTs using Minimum Description Length (MDL).")
        learner.useMDLCorrection()
        learned_bn = learner.learnParameters(bn.dag())
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Please use 'MLE', 'EM', or 'MDL'.")

    # Extract learned CPTs for relevant nodes and format them using string labels
    for node in bn.names():
        cpt = learned_bn.cpt(node)
        print(f"[INFO] Learned CPT for node '{node}' has shape {cpt.shape}.")
        
        # Adjust the learned CPT shape if necessary
        learned_cpts[node] = [[prob for prob in row] for row in cpt]

    print(f"[INFO] Learned CPTs for {algorithm} generated successfully.")
    return learned_cpts



# Unified function for network structure creation
def create_standard_network():
    """
    Creates a consistent network structure to be used for Value, Quality, and Investment Networks.
    
    Returns
    -------
    bn: gum.BayesNet
        The defined Bayesian Network with standard nodes and arcs.
    """
    bn = gum.BayesNet("StandardNetwork")

    # Define nodes
    value_relative_to_price = gum.LabelizedVariable('ValueRelativeToPrice', 'Value Relative to Price', 3)
    value_relative_to_price.changeLabel(0, 'Cheap')
    value_relative_to_price.changeLabel(1, 'FairValue')
    value_relative_to_price.changeLabel(2, 'Expensive')
    bn.add(value_relative_to_price)

    pe_relative_market = gum.LabelizedVariable('PERelative_ShareMarket', 'PE Relative to Market', 3)
    pe_relative_market.changeLabel(0, 'Cheap')
    pe_relative_market.changeLabel(1, 'FairValue')
    pe_relative_market.changeLabel(2, 'Expensive')
    bn.add(pe_relative_market)

    pe_relative_sector = gum.LabelizedVariable('PERelative_ShareSector', 'PE Relative to Sector', 3)
    pe_relative_sector.changeLabel(0, 'Cheap')
    pe_relative_sector.changeLabel(1, 'FairValue')
    pe_relative_sector.changeLabel(2, 'Expensive')
    bn.add(pe_relative_sector)

    forward_pe_current_vs_history = gum.LabelizedVariable('ForwardPE_CurrentVsHistory', 'Forward PE vs History', 3)
    forward_pe_current_vs_history.changeLabel(0, 'Cheap')
    forward_pe_current_vs_history.changeLabel(1, 'FairValue')
    forward_pe_current_vs_history.changeLabel(2, 'Expensive')
    bn.add(forward_pe_current_vs_history)

    future_share_performance = gum.LabelizedVariable('FutureSharePerformance', 'Future Performance', 3)
    future_share_performance.changeLabel(0, 'Positive')
    future_share_performance.changeLabel(1, 'Stagnant')
    future_share_performance.changeLabel(2, 'Negative')
    bn.add(future_share_performance)

    quality = gum.LabelizedVariable('Quality', 'Quality of Investment', 3)
    quality.changeLabel(0, 'High')
    quality.changeLabel(1, 'Medium')
    quality.changeLabel(2, 'Low')
    bn.add(quality)

    investable = gum.LabelizedVariable('Investable', 'Investable share', 2)
    investable.changeLabel(0, 'Yes')
    investable.changeLabel(1, 'No')
    bn.add(investable)

    print("[INFO] Nodes and labels successfully added to the Standard Network.")

    # Define arcs
    bn.addArc(bn.idFromName('FutureSharePerformance'), bn.idFromName('PERelative_ShareMarket'))
    bn.addArc(bn.idFromName('FutureSharePerformance'), bn.idFromName('PERelative_ShareSector'))
    bn.addArc(bn.idFromName('FutureSharePerformance'), bn.idFromName('ForwardPE_CurrentVsHistory'))
    bn.addArc(bn.idFromName('PERelative_ShareMarket'), bn.idFromName('ValueRelativeToPrice'))
    bn.addArc(bn.idFromName('PERelative_ShareSector'), bn.idFromName('ValueRelativeToPrice'))
    bn.addArc(bn.idFromName('ForwardPE_CurrentVsHistory'), bn.idFromName('ValueRelativeToPrice'))

    print("[INFO] Arcs successfully added to the Standard Network.")
    
    return bn

# Updated Value Network
def value_network(pe_relative_market_state, pe_relative_sector_state, forward_pe_current_vs_history_state,
                  future_performance_state=None, learned_cpts=None):
    """
    Returns the final Value Network decision, with learned CPTs if provided.
    """
    print("[INFO] Initializing the Value Network Influence Diagram.")
    ve_model = create_standard_network()

    # Update CPT values if learned CPTs are provided
    if learned_cpts:
        update_cpt_values(ve_model, learned_cpts)

    # Create the inference engine and run inference
    ie = gum.ShaferShenoyLIMIDInference(ve_model)
    ie.addNoForgettingAssumption(['ValueRelativeToPrice'])

    print(f"[INFO] Evidence added. Running inference on Value Network...")
    ie.makeInference()
    print("[INFO] Inference completed successfully.")

    # Extract and return the final decision
    decision_index = int(np.argmax(ie.posteriorUtility('ValueRelativeToPrice').toarray()))  # Ensure decision_index is an int
    decision = ve_model.variable('ValueRelativeToPrice').label(decision_index)

    print(f"[INFO] Final Value Network decision: {decision}")
    return decision

# Updated Quality Network
def quality_network(roe_vs_coe_state, relative_debt_equity_state, cagr_vs_inflation_state,
                    systematic_risk_state, extension=False, learned_cpts=None):
    """
    Returns the final Quality Network decision, with learned CPTs if provided.
    """
    print("[INFO] Initializing the Quality Network Bayesian Decision Network.")
    qn_model = create_standard_network()  # Use the standard network structure

    # Update CPT values if learned CPTs are provided
    if learned_cpts:
        update_cpt_values(qn_model, learned_cpts)

    # Inference
    ie = gum.ShaferShenoyLIMIDInference(qn_model)
    ie.addNoForgettingAssumption(['Quality'])
    ie.makeInference()

    # Extract the decision
    decision_index = int(np.argmax(ie.posteriorUtility('Quality').toarray()))  # Ensure int type
    decision = qn_model.variable('Quality').label(decision_index)

    print(f"[INFO] Final Quality Network decision: {decision}")
    return decision

# Updated Investment Recommendation Network
def investment_recommendation(value_decision, quality_decision, learned_cpts=None):
    """
    Returns the final Investment Recommendation with learned CPTs if provided.
    """
    print(f"[INFO] Creating Investment Recommendation Influence Diagram.")
    ir_model = create_standard_network()  # Use the standard network structure

    # Update CPT values if learned CPTs are provided
    if learned_cpts:
        update_cpt_values(ir_model, learned_cpts)

    # Create the inference engine
    print(f"[INFO] Setting up inference engine.")
    ie = gum.ShaferShenoyLIMIDInference(ir_model)

    # Add evidence based on value and quality decisions
    if value_decision == "Cheap":
        ie.addEvidence('Value', [1, 0, 0])
    elif value_decision == "FairValue":
        ie.addEvidence('Value', [0, 1, 0])
    else:
        ie.addEvidence('Value', [0, 0, 1])

    if quality_decision == "High":
        ie.addEvidence('Quality', [1, 0, 0])
    elif quality_decision == "Medium":
        ie.addEvidence('Quality', [0, 1, 0])
    else:
        ie.addEvidence('Quality', [0, 0, 1])

    print(f"[INFO] Evidence added for Value and Quality nodes. Running inference...")
    ie.makeInference()
    print("[INFO] Inference completed successfully.")

    # Extract and return the final decision
    decision_index = np.argmax(ie.posteriorUtility('Investable').toarray())
    decision = ir_model.variable('Investable').label(int(decision_index))

    print(f"[INFO] Final decision for Investable Network: {decision}")
    return decision

# Update CPT Function
def update_cpt_values(network, learned_cpts):
    """
    Updates the network with learned CPTs.
    """
    print("[INFO] Updating CPTs with learned values...")
    for node_name, cpt_values in learned_cpts.items():
        print(f"[INFO] Attempting to update node '{node_name}' with learned CPT.")
        if node_name in network.names():
            try:
                expected_shape = network.cpt(network.idFromName(node_name)).shape
                reshaped_cpt = np.reshape(cpt_values, expected_shape)
                network.cpt(network.idFromName(node_name)).fillWith(reshaped_cpt)
                print(f"[INFO] Successfully updated CPT for node '{node_name}'.")
            except Exception as e:
                print(f"[ERROR] Could not update CPT for node '{node_name}': {e}")
        else:
            print(f"[WARNING] Node '{node_name}' not found in the Network. Skipping...")
