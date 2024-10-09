import os
import numpy as np
import pyAgrum as gum

def investment_recommendation(value_decision, quality_decision, learned_cpts=None):
    """
    Returns the final Investment Recommendation with learned CPTs if provided.

    Parameters
    ----------
    value_decision : str
       Final decision output of the Value Network.
    quality_decision : str
       Final decision output of the Quality Network.
    learned_cpts: dict, optional
        Dictionary of learned CPTs for updating the Bayesian Network.

    Returns
    -------
    str : Final investment decision (Yes or No).
    """
    print(f"[INFO] Creating Investment Recommendation Influence Diagram.")
    ir_model = gum.InfluenceDiagram()

    # Define the nodes
    investable = gum.LabelizedVariable('Investable', 'Investable share', 2)
    investable.changeLabel(0, 'Yes')
    investable.changeLabel(1, 'No')
    ir_model.addDecisionNode(investable)

    share_performance = gum.LabelizedVariable('Performance', '', 3)
    share_performance.changeLabel(0, 'Positive')
    share_performance.changeLabel(1, 'Stagnant')
    share_performance.changeLabel(2, 'Negative')
    ir_model.addChanceNode(share_performance)

    value = gum.LabelizedVariable('Value', 'Value', 3)
    value.changeLabel(0, 'Cheap')
    value.changeLabel(1, 'FairValue')
    value.changeLabel(2, 'Expensive')
    ir_model.addChanceNode(value)

    quality = gum.LabelizedVariable('Quality', 'Quality', 3)
    quality.changeLabel(0, 'High')
    quality.changeLabel(1, 'Medium')
    quality.changeLabel(2, 'Low')
    ir_model.addChanceNode(quality)

    investment_utility = gum.LabelizedVariable('I_Utility', '', 1)
    ir_model.addUtilityNode(investment_utility)

    # Define arcs
    ir_model.addArc(ir_model.idFromName('Performance'), ir_model.idFromName('Quality'))
    ir_model.addArc(ir_model.idFromName('Performance'), ir_model.idFromName('Value'))
    ir_model.addArc(ir_model.idFromName('Performance'), ir_model.idFromName('I_Utility'))
    ir_model.addArc(ir_model.idFromName('Value'), ir_model.idFromName('Investable'))
    ir_model.addArc(ir_model.idFromName('Quality'), ir_model.idFromName('Investable'))
    ir_model.addArc(ir_model.idFromName('Investable'), ir_model.idFromName('I_Utility'))

    print(f"[INFO] Influence Diagram created with nodes: {ir_model.names()}")

    # Define Utility Values
    ir_model.utility(ir_model.idFromName('I_Utility'))[{'Investable': 'Yes'}] = [[300], [-100], [-250]]
    ir_model.utility(ir_model.idFromName('I_Utility'))[{'Investable': 'No'}] = [[-200], [100], [200]]

    # Update CPTs if learned CPTs are provided
    if learned_cpts:
        print("[INFO] Updating CPTs with learned values.")
        for node_name, cpt_values in learned_cpts.items():
            if node_name in ir_model.names():
                ir_model.cpt(ir_model.idFromName(node_name)).fillWith(cpt_values)
                print(f"[INFO] Updated CPT for node '{node_name}'.")
            else:
                print(f"[WARNING] Node '{node_name}' not found in the Influence Diagram. Skipping...")

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
    var = ie.posteriorUtility('Investable').variable('Investable')
    decision_index = np.argmax(ie.posteriorUtility('Investable').toarray())
    decision = var.label(int(decision_index))

    print(f"[INFO] Final decision for Investable Network: {decision}")
    return format(decision)
