"""Frozen AB1 corner/control ordering. No labels and no model execution."""
from ab1_boundary_core import corner, control
EDGES={"Q_R_E":("H00","H10","Q"),"Q_R_B":("H11","H01","Q"),"R_Q_E":("H00","H01","R"),"R_Q_B":("H11","H10","R")}
def build_boundary_endpoints(E,B,prefix,R,*,split,ident):
 H={"H00":corner(E,B,prefix,R,"E","E"),"H10":corner(E,B,prefix,R,"B","E"),"H01":corner(E,B,prefix,R,"E","B"),"H11":corner(E,B,prefix,R,"B","B")};out={k:{"status":"OK","input":v} for k,v in H.items()}
 for edge,(start,end,region) in EDGES.items():
  for kind in ("norm","random"):
   x,r=control(H[start],H[end],prefix,R,region,kind,split,ident,edge);out[f"{edge}_{kind}"]={"status":r["status"],"input":x,"receipt":r,"receiver":start,"donor":end}
 assert len(out)==12 and set(H)=={"H00","H10","H01","H11"};return out
