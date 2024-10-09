import pyAgrum as gum
import numpy as np

def quality_network(roe_vs_coe_state, relative_debt_equity_state, cagr_vs_inflation_state,
                    systematic_risk_state, extension=False, learned_cpts=None):
    """
    Returns the final Quality Network decision, with learned CPTs if provided.

    Parameters
    ----------
    roe_vs_coe_state : str
       Discrete state for ROE vs COE.
    relative_debt_equity_state : str
       Discrete state for Relative Debt to Equity.
    cagr_vs_inflation_state : str
        Discrete state for CAGR vs Inflation.
    systematic_risk_state : str
        Discrete state for Systematic Risk.
    extension: bool, optional
        If True, include the systematic risk extension.
    learned_cpts: dict, optional
        Dictionary of learned CPTs for updating the Bayesian Network.

    Returns
    -------
    str : Final decision for the Quality Network (High, Medium, Low).
    """
    print("[INFO] Initializing the Quality Network Bayesian Decision Network.")
    
    # Define an Influence Diagram instead of a BayesNet
    qn_model = gum.InfluenceDiagram()

    # Define nodes
    roe_vs_coe_node = gum.LabelizedVariable('ROE_vs_COE', 'ROE vs COE', 3)
    roe_vs_coe_node.changeLabel(0, 'Low')
    roe_vs_coe_node.changeLabel(1, 'Medium')
    roe_vs_coe_node.changeLabel(2, 'High')
    qn_model.addChanceNode(roe_vs_coe_node)

    relative_debt_equity_node = gum.LabelizedVariable('Relative_Debt_Equity', 'Relative Debt to Equity', 3)
    relative_debt_equity_node.changeLabel(0, 'Low')
    relative_debt_equity_node.changeLabel(1, 'Medium')
    relative_debt_equity_node.changeLabel(2, 'High')
    qn_model.addChanceNode(relative_debt_equity_node)

    cagr_vs_inflation_node = gum.LabelizedVariable('CAGR_vs_Inflation', 'CAGR vs Inflation', 3)
    cagr_vs_inflation_node.changeLabel(0, 'Low')
    cagr_vs_inflation_node.changeLabel(1, 'Medium')
    cagr_vs_inflation_node.changeLabel(2, 'High')
    qn_model.addChanceNode(cagr_vs_inflation_node)

    systematic_risk_node = gum.LabelizedVariable('Systematic_Risk', 'Systematic Risk', 3)
    systematic_risk_node.changeLabel(0, 'Low')
    systematic_risk_node.changeLabel(1, 'Medium')
    systematic_risk_node.changeLabel(2, 'High')
    qn_model.addChanceNode(systematic_risk_node)

    # Decision node for Quality
    quality_node = gum.LabelizedVariable('Quality', 'Quality of Investment', 3)
    quality_node.changeLabel(0, 'Low')
    quality_node.changeLabel(1, 'Medium')
    quality_node.changeLabel(2, 'High')
    qn_model.addDecisionNode(quality_node)

    print("[INFO] Nodes successfully added to the Quality Network.")

    # Define arcs
    qn_model.addArc(qn_model.idFromName('ROE_vs_COE'), qn_model.idFromName('Quality'))
    qn_model.addArc(qn_model.idFromName('Relative_Debt_Equity'), qn_model.idFromName('Quality'))
    qn_model.addArc(qn_model.idFromName('CAGR_vs_Inflation'), qn_model.idFromName('Quality'))
    if extension:
        qn_model.addArc(qn_model.idFromName('Systematic_Risk'), qn_model.idFromName('Quality'))

    print("[INFO] Arcs successfully added to the Quality Network.")

    # Update CPTs if learned CPTs are provided
    if learned_cpts:
        print("[INFO] Updating CPTs with learned values in the Quality Network...")
        print(f"[INFO] Available nodes in the Quality Network: {qn_model.names()}")
        for node_name, cpt_values in learned_cpts.items():
            print(f"[INFO] Attempting to update node '{node_name}' with learned CPT.")
            if node_name in qn_model.names():
                try:
                    expected_shape = qn_model.cpt(qn_model.idFromName(node_name)).shape
                    print(f"[INFO] Expected CPT shape for node '{node_name}': {expected_shape}")

                    # Adjust CPT values to match the expected shape
                    reshaped_cpt = np.reshape(cpt_values, expected_shape)
                    qn_model.cpt(qn_model.idFromName(node_name)).fillWith(reshaped_cpt)
                    print(f"[INFO] Successfully updated CPT for node '{node_name}'.")

                except ValueError as ve:
                    print(f"[ERROR] CPT shape mismatch for node '{node_name}': {ve}")
                except Exception as e:
                    print(f"[ERROR] Could not update CPT for node '{node_name}': {e}")
            else:
                print(f"[WARNING] Node '{node_name}' not found in the Quality Network. Skipping...")

    # Inference
    ie = gum.ShaferShenoyLIMIDInference(qn_model)
    ie.addNoForgettingAssumption(['Quality'])
    ie.makeInference()

    # Extract the decision
    decision_index = int(np.argmax(ie.posteriorUtility('Quality').toarray()))  # Ensure int type
    decision = qn_model.variable('Quality').label(decision_index)

    print(f"[INFO] Final Quality Network decision: {decision}")
    return decision