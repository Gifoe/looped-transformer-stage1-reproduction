from round2_intervention_analysis import factorial_metrics
x=factorial_metrics(10,7,6,2)
assert x['context_contribution_at_R2']==3 and x['context_contribution_at_R1']==4 and x['readout_contribution_at_C2']==4 and x['readout_contribution_at_C1']==5 and x['interaction']==-1
assert x['context_replacement_K2_to_K1_at_R2']==-3 and x['readout_replacement_K2_to_K1_at_C2']==-4
print('PASS fixed factorial direction and interaction')
