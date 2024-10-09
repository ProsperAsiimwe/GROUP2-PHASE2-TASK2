import os
import numpy as np
import pyAgrum as gum

def value_network(pe_relative_market_state, pe_relative_sector_state, forward_pe_current_vs_history_state,
                  future_performance_state=None, learned_cpts=None):
    """
    Returns the final Value Network decision, with learned CPTs if provided.

    Parameters
    ----------
    pe_relative_market_state : str
       Discrete state for PE relative to market.
    pe_relative_sector_state : str
       Discrete state for PE relative to sector.
    forward_pe_current_vs_history_state: str
        Discrete state for Forward PE Current vs History.
    future_performance_state: Union[None, str]
        Default value is None.
    learned_cpts: dict, optional
        Dictionary of learned CPTs for updating the Bayesian Network.

    Returns
    -------
    str : Final decision for the Value Network (Cheap, FairValue, Expensive).
    """
    print("[INFO] Initializing the Value Network Influence Diagram.")
    ve_model = gum.InfluenceDiagram()

    # Define decision nodes
    expensive_decision = gum.LabelizedVariable('Expensive_E', 'Is Expensive', 2)
    expensive_decision.changeLabel(0, 'No')
    expensive_decision.changeLabel(1, 'Yes')
    ve_model.addDecisionNode(expensive_decision)

    value_relative_to_price_decision = gum.LabelizedVariable('ValueRelativeToPrice', '', 3)
    value_relative_to_price_decision.changeLabel(0, 'Cheap')
    value_relative_to_price_decision.changeLabel(1, 'FairValue')
    value_relative_to_price_decision.changeLabel(2, 'Expensive')
    ve_model.addDecisionNode(value_relative_to_price_decision)

    # Define chance nodes
    future_share_performance = gum.LabelizedVariable('FutureSharePerformance', 'Future Performance', 3)
    future_share_performance.changeLabel(0, 'Positive')
    future_share_performance.changeLabel(1, 'Stagnant')
    future_share_performance.changeLabel(2, 'Negative')
    ve_model.addChanceNode(future_share_performance)

    pe_relative_market = gum.LabelizedVariable('PERelative_ShareMarket', 'PE Relative to Market', 3)
    pe_relative_market.changeLabel(0, 'Cheap')
    pe_relative_market.changeLabel(1, 'FairValue')
    pe_relative_market.changeLabel(2, 'Expensive')
    ve_model.addChanceNode(pe_relative_market)

    pe_relative_sector = gum.LabelizedVariable('PERelative_ShareSector', 'PE Relative to Sector', 3)
    pe_relative_sector.changeLabel(0, 'Cheap')
    pe_relative_sector.changeLabel(1, 'FairValue')
    pe_relative_sector.changeLabel(2, 'Expensive')
    ve_model.addChanceNode(pe_relative_sector)

    forward_pe_current_vs_history = gum.LabelizedVariable('ForwardPE_CurrentVsHistory', 'Forward PE vs History', 3)
    forward_pe_current_vs_history.changeLabel(0, 'Cheap')
    forward_pe_current_vs_history.changeLabel(1, 'FairValue')
    forward_pe_current_vs_history.changeLabel(2, 'Expensive')
    ve_model.addChanceNode(forward_pe_current_vs_history)

    # Define utility nodes
    utility_expensive = gum.LabelizedVariable('Expensive_Utility', 'Expensive Utility', 1)
    ve_model.addUtilityNode(utility_expensive)

    utility_value_relative_to_price = gum.LabelizedVariable('VRP_Utility', 'Value Relative to Price Utility', 1)
    ve_model.addUtilityNode(utility_value_relative_to_price)

    print(f"[INFO] Nodes added to the Value Network: {ve_model.names()}")

    # Define arcs (dependencies)
    ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName('PERelative_ShareMarket'))
    ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName('PERelative_ShareSector'))
    ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName('ForwardPE_CurrentVsHistory'))
    ve_model.addArc(ve_model.idFromName('PERelative_ShareMarket'), ve_model.idFromName('Expensive_E'))
    ve_model.addArc(ve_model.idFromName('PERelative_ShareSector'), ve_model.idFromName('Expensive_E'))
    ve_model.addArc(ve_model.idFromName('ForwardPE_CurrentVsHistory'), ve_model.idFromName('ValueRelativeToPrice'))

    print(f"[INFO] Arcs added to the Value Network.")

    # Update CPTs if learned CPTs are provided
    if learned_cpts:
        print("[INFO] Updating CPTs with learned values.")
        for node_name, cpt_values in learned_cpts.items():
            if node_name in ve_model.names():
                ve_model.cpt(ve_model.idFromName(node_name)).fillWith(cpt_values)
                print(f"[INFO] Updated CPT for node '{node_name}'.")
            else:
                print(f"[WARNING] Node '{node_name}' not found in the Value Network. Skipping...")

    # Create the inference engine
    ie = gum.ShaferShenoyLIMIDInference(ve_model)
    ie.addNoForgettingAssumption(['Expensive_E', 'ValueRelativeToPrice'])

    # Set evidence based on input states
    print(f"[INFO] Adding evidence for PE and Forward PE states.")
    if pe_relative_market_state == "cheap":
        ie.addEvidence('PERelative_ShareMarket', [1, 0, 0])
    elif pe_relative_market_state == "fairValue":
        ie.addEvidence('PERelative_ShareMarket', [0, 1, 0])
    else:
        ie.addEvidence('PERelative_ShareMarket', [0, 0, 1])

    if pe_relative_sector_state == "cheap":
        ie.addEvidence('PERelative_ShareSector', [1, 0, 0])
    elif pe_relative_sector_state == "fairValue":
        ie.addEvidence('PERelative_ShareSector', [0, 1, 0])
    else:
        ie.addEvidence('PERelative_ShareSector', [0, 0, 1])

    if forward_pe_current_vs_history_state == "cheap":
        ie.addEvidence('ForwardPE_CurrentVsHistory', [1, 0, 0])
    elif forward_pe_current_vs_history_state == "fairValue":
        ie.addEvidence('ForwardPE_CurrentVsHistory', [0, 1, 0])
    else:
        ie.addEvidence('ForwardPE_CurrentVsHistory', [0, 0, 1])

    if future_performance_state:
        if future_performance_state.lower() in ["positive", "1"]:
            ie.addEvidence('FutureSharePerformance', [0.8, 0.1, 0.1])
        elif future_performance_state.lower() in ["stagnant", "0"]:
            ie.addEvidence('FutureSharePerformance', [0.1, 0.2, 0.1])
        else:
            ie.addEvidence('FutureSharePerformance', [0.1, 0.1, 0.8])

    print(f"[INFO] Evidence added. Running inference on Value Network...")
    ie.makeInference()
    print("[INFO] Inference completed successfully.")

    # Extract and return the final decision
    var = ie.posteriorUtility('ValueRelativeToPrice').variable('ValueRelativeToPrice')
    decision_index = np.argmax(ie.posteriorUtility('ValueRelativeToPrice').toarray())
    decision = var.label(int(decision_index))

    print(f"[INFO] Final Value Network decision: {decision}")
    return format(decision)
