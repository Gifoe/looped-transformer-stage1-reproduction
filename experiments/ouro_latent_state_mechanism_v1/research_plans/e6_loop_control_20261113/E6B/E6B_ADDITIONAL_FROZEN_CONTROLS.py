"""Paired comparisons to existing, already-frozen E6A depth/confidence controls. No selection."""
import argparse,json,importlib.util
from pathlib import Path
from E6B_ANALYSIS import controller_metrics,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--decisions',required=True);p.add_argument('--labels',required=True);p.add_argument('--e6a-methods',required=True);p.add_argument('--e6a-spec',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 spec=json.loads(Path(a.e6a_spec).read_text());im=importlib.util.spec_from_file_location('frozen_a',a.e6a_methods);m=importlib.util.module_from_spec(im);im.loader.exec_module(m)
 labels={(r['split'],r['id']):r['gold'] for r in map(json.loads,Path(a.labels).read_text().splitlines())};features={(r['split'],r['id']):r for r in map(json.loads,Path(a.features).read_text().splitlines())};rows=[r for r in map(json.loads,Path(a.decisions).read_text().splitlines()) if r['split']=='test'];out=[]
 for name in ['highest_margin','highest_entropy','early_fixed']:
  assert name in spec['methods'];baseline={key:m.infer(r['native_scores'],spec['methods'][name]) for key,r in features.items() if key[0]=='test'}
  for i in sorted({r['method_index'] for r in rows}):
   rs=[r for r in rows if r['method_index']==i];gold=[labels[(r['split'],r['id'])] for r in rs];pred=[r['prediction'] for r in rs];base=[baseline[(r['split'],r['id'])]['prediction'] for r in rs];cost=[r['executed_loops'] for r in rs]
   v=controller_metrics([{'gold':g} for g in gold],rs,base);v.update(method_index=i,comparator=name,comparator_correct=sum(x==y for x,y in zip(base,gold)));out.append(v)
 Path(a.out).write_text(json.dumps({'source_sha256':sha(__file__),'e6a_frozen_spec_sha256':sha(a.e6a_spec),'scope':'All existing fixed highest-confidence and early-depth controls; no new method or Test selection.','comparisons':out},indent=2))
 print('ADDITIONAL_FROZEN_CONTROLS_COMPLETE',len(out))
if __name__=='__main__':main()
