import pyAgrum as gum
import numpy as np

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

    # Define decision and chance nodes
    value_relative_to_price = gum.LabelizedVariable('ValueRelativeToPrice', 'Value Relative to Price', 3)
    value_relative_to_price.changeLabel(0, 'Cheap')
    value_relative_to_price.changeLabel(1, 'FairValue')
    value_relative_to_price.changeLabel(2, 'Expensive')
    ve_model.addDecisionNode(value_relative_to_price)

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

    future_share_performance = gum.LabelizedVariable('FutureSharePerformance', 'Future Performance', 3)
    future_share_performance.changeLabel(0, 'Positive')
    future_share_performance.changeLabel(1, 'Stagnant')
    future_share_performance.changeLabel(2, 'Negative')
    ve_model.addChanceNode(future_share_performance)

    forward_pe_current_vs_history = gum.LabelizedVariable('ForwardPE_CurrentVsHistory', 'Forward PE vs History', 3)
    forward_pe_current_vs_history.changeLabel(0, 'Cheap')
    forward_pe_current_vs_history.changeLabel(1, 'FairValue')
    forward_pe_current_vs_history.changeLabel(2, 'Expensive')
    ve_model.addChanceNode(forward_pe_current_vs_history)

    print(f"[INFO] Nodes and labels successfully added to the Value Network.")

    # Add arcs to represent dependencies
    ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName('PERelative_ShareMarket'))
    ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName('PERelative_ShareSector'))
    ve_model.addArc(ve_model.idFromName('FutureSharePerformance'), ve_model.idFromName('ForwardPE_CurrentVsHistory'))
    ve_model.addArc(ve_model.idFromName('PERelative_ShareMarket'), ve_model.idFromName('ValueRelativeToPrice'))
    ve_model.addArc(ve_model.idFromName('PERelative_ShareSector'), ve_model.idFromName('ValueRelativeToPrice'))
    ve_model.addArc(ve_model.idFromName('ForwardPE_CurrentVsHistory'), ve_model.idFromName('ValueRelativeToPrice'))

    print(f"[INFO] Arcs successfully added to the Value Network.")

    # Debug: Print learned CPTs
    if learned_cpts:
        print("[INFO] Updating CPTs with learned values...")
        print(f"[INFO] Available nodes in the Value Network: {ve_model.names()}")
        for node_name, cpt_values in learned_cpts.items():
            print(f"[INFO] Attempting to update node '{node_name}' with learned CPT.")
            if node_name in ve_model.names():
                try:
                    # Retrieve expected shape from the node's CPT
                    expected_shape = ve_model.cpt(node_name).shape
                    print(f"[INFO] Expected CPT shape for node '{node_name}': {expected_shape}")
                    
                    # Adjust CPT values to match the expected shape
                    reshaped_cpt = np.reshape(cpt_values, expected_shape)
                    
                    # Update the CPT with the new values
                    ve_model.cpt(node_name).fillWith(reshaped_cpt)
                    print(f"[INFO] Successfully updated CPT for node '{node_name}'.")
                    
                except ValueError as ve:
                    print(f"[ERROR] CPT shape mismatch for node '{node_name}': {ve}")
                except Exception as e:
                    print(f"[ERROR] Could not update CPT for node '{node_name}': {e}")
            else:
                print(f"[WARNING] Node '{node_name}' not found in the Value Network. Skipping...")

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