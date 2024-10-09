import os
import numpy as np
import pyAgrum as gum

def quality_network(roe_vs_coe, debt_equity, cagr_vs_inflation, systematic_risk, extension=False, learned_cpts=None):
    """
    Returns the final Quality Network decision, with learned CPTs if provided.

    Parameters
    ----------
    roe_vs_coe : str
        Discrete state for Return on Equity (ROE) vs Cost of Equity (COE).
    debt_equity : str
        Discrete state for Debt to Equity ratio.
    cagr_vs_inflation: str
        Discrete state for Compound Annual Growth Rate (CAGR) vs Inflation.
    systematic_risk: str
        Discrete state for systematic risk.
    extension : bool, optional
        Use systematic risk extension, default is False.
    learned_cpts: dict, optional
        Dictionary of learned CPTs for updating the Bayesian Network.

    Returns
    -------
    str : Final decision for the Quality Network (High, Medium, Low).
    """
    print("[INFO] Initializing the Quality Network Bayesian Decision Network.")
    qn_model = gum.BayesNet("Quality Network")

    # Define chance nodes
    roe_vs_coe_node = gum.LabelizedVariable('ROE_vs_COE', 'Return on Equity vs Cost of Equity', 3)
    roe_vs_coe_node.changeLabel(0, 'Below')
    roe_vs_coe_node.changeLabel(1, 'Equal')
    roe_vs_coe_node.changeLabel(2, 'Above')
    qn_model.addChanceNode(roe_vs_coe_node)

    debt_equity_node = gum.LabelizedVariable('Relative_Debt_Equity', 'Debt/Equity', 3)
    debt_equity_node.changeLabel(0, 'Low')
    debt_equity_node.changeLabel(1, 'Moderate')
    debt_equity_node.changeLabel(2, 'High')
    qn_model.addChanceNode(debt_equity_node)

    cagr_vs_inflation_node = gum.LabelizedVariable('CAGR_vs_Inflation', 'CAGR vs Inflation', 3)
    cagr_vs_inflation_node.changeLabel(0, 'Below')
    cagr_vs_inflation_node.changeLabel(1, 'Equal')
    cagr_vs_inflation_node.changeLabel(2, 'Above')
    qn_model.addChanceNode(cagr_vs_inflation_node)

    systematic_risk_node = gum.LabelizedVariable('Systematic_Risk', 'Systematic Risk', 3)
    systematic_risk_node.changeLabel(0, 'Low')
    systematic_risk_node.changeLabel(1, 'Medium')
    systematic_risk_node.changeLabel(2, 'High')
    qn_model.addChanceNode(systematic_risk_node)

    quality = gum.LabelizedVariable('Quality', 'Company Quality', 3)
    quality.changeLabel(0, 'Low')
    quality.changeLabel(1, 'Medium')
    quality.changeLabel(2, 'High')
    qn_model.addChanceNode(quality)

    print(f"[INFO] Nodes added to the Quality Network: {qn_model.names()}")

    # Define arcs (dependencies)
    qn_model.addArc(qn_model.idFromName('ROE_vs_COE'), qn_model.idFromName('Quality'))
    qn_model.addArc(qn_model.idFromName('Relative_Debt_Equity'), qn_model.idFromName('Quality'))
    qn_model.addArc(qn_model.idFromName('CAGR_vs_Inflation'), qn_model.idFromName('Quality'))
    qn_model.addArc(qn_model.idFromName('Systematic_Risk'), qn_model.idFromName('Quality'))

    print(f"[INFO] Arcs added to the Quality Network.")

    # Update CPTs if learned CPTs are provided
    if learned_cpts:
        print("[INFO] Updating CPTs with learned values.")
        for node_name, cpt_values in learned_cpts.items():
            if node_name in qn_model.names():
                qn_model.cpt(qn_model.idFromName(node_name)).fillWith(cpt_values)
                print(f"[INFO] Updated CPT for node '{node_name}'.")
            else:
                print(f"[WARNING] Node '{node_name}' not found in the Quality Network. Skipping...")

    # Create the inference engine
    print(f"[INFO] Setting up inference engine for the Quality Network.")
    ie = gum.LazyPropagation(qn_model)

    # Set evidence for each input parameter
    print(f"[INFO] Adding evidence for ROE vs COE, Debt/Equity, CAGR vs Inflation, and Systematic Risk.")
    if roe_vs_coe == "below":
        ie.addEvidence('ROE_vs_COE', [1, 0, 0])
    elif roe_vs_coe == "equal":
        ie.addEvidence('ROE_vs_COE', [0, 1, 0])
    else:
        ie.addEvidence('ROE_vs_COE', [0, 0, 1])

    if debt_equity == "low":
        ie.addEvidence('Relative_Debt_Equity', [1, 0, 0])
    elif debt_equity == "moderate":
        ie.addEvidence('Relative_Debt_Equity', [0, 1, 0])
    else:
        ie.addEvidence('Relative_Debt_Equity', [0, 0, 1])

    if cagr_vs_inflation == "below":
        ie.addEvidence('CAGR_vs_Inflation', [1, 0, 0])
    elif cagr_vs_inflation == "equal":
        ie.addEvidence('CAGR_vs_Inflation', [0, 1, 0])
    else:
        ie.addEvidence('CAGR_vs_Inflation', [0, 0, 1])

    if systematic_risk == "low":
        ie.addEvidence('Systematic_Risk', [1, 0, 0])
    elif systematic_risk == "medium":
        ie.addEvidence('Systematic_Risk', [0, 1, 0])
    else:
        ie.addEvidence('Systematic_Risk', [0, 0, 1])

    print(f"[INFO] Evidence added. Running inference on Quality Network...")
    ie.makeInference()
    print("[INFO] Inference completed successfully.")

    # Extract and return the final decision
    var = ie.posterior("Quality").variable("Quality")
    decision_index = np.argmax(ie.posterior("Quality").tolist())
    decision = var.label(int(decision_index))

    print(f"[INFO] Final Quality Network decision: {decision}")
    return format(decision)
