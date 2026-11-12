from behavior_summary import summarize
r=[{'gold':0,'predictions':p} for p in ([1,0,0,0],[0,1,0,1],[1,1,1,1],[0,0,0,0])]
x=summarize(r);assert x['correct_counts']==[2,2,3,2];assert [(z['rescue'],z['damage']) for z in x['transitions']]==[(1,1),(1,0),(0,1)];assert x['first_transition_share_of_total_net'] is None;assert summarize([])['n']==0
y=summarize([{'gold':0,'predictions':[1,0,0,0],'top_score_ties':[False,True,False,False],'any_score_ties':[False,True,False,False]},{'gold':0,'predictions':[1,0,1,1]},{'gold':0,'predictions':[1,1,1,1]}]);assert y['total_net_gain']==1 and y['first_transition_net_gain']==2 and y['first_transition_share_of_total_net']==2;assert y['top_score_ties']==[0,1,0,0] and y['any_score_ties']==[0,1,0,0]
z=summarize([{'gold':0,'predictions':[0,0,1,1]}]);assert z['total_net_gain']==-1 and z['first_transition_share_of_total_net'] is None
print('PASS trajectories, rescue damage, zero net, later damage ratio, ties, empty subgroup')
