"""Engineering-only AST adapter: group partition + per-shard completion filename.
Frozen intervention, data, scoring and gates are executed unchanged.
"""
import ast,pathlib,json,hashlib,sys,os,datetime
from freeze_guard import verify
R=pathlib.Path('/root/ouro-semantic-read-sol-20261117');fsha=verify();stage,phase,rank,n=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]);assert stage in ['native','science'] and phase=='confirmation' and n==3 and 0<=rank<n;source=R/'code'/('run_native.py' if stage=='native' else 'run_science.py');original=source.read_text();tree=ast.parse(original)
def select_shard(rows):return [q for q in rows if int(q['pair_id'][2:])%n==rank]
class Adapter(ast.NodeTransformer):
 def __init__(self):self.filters=0;self.names=0
 def visit_Assign(self,node):
  node=self.generic_visit(node)
  if any(isinstance(t,ast.Name) and t.id=='qs' for t in node.targets):node.value=ast.Call(func=ast.Name(id='select_shard',ctx=ast.Load()),args=[node.value],keywords=[]);self.filters+=1
  return node
 def visit_Constant(self,node):
  expected='_NATIVE_COMPLETE.json' if stage=='native' else '_SCIENCE_COMPLETE.json'
  if node.value==expected:self.names+=1;return ast.copy_location(ast.Constant(expected.replace('_COMPLETE',f'_SHARD{rank}OF{n}_COMPLETE')),node)
  return node
adapter=Adapter();modified=ast.fix_missing_locations(adapter.visit(tree));assert adapter.filters==adapter.names==1;actual=ast.unparse(modified);receipt={'freeze_sha256':fsha,'stage':stage,'phase':phase,'rank':rank,'workers':n,'pid':os.getpid(),'original_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'executed_AST_source_sha256':hashlib.sha256(actual.encode()).hexdigest(),'wrapper_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'transformations':['filtercompletebasegroupsbyintegeridmodulo3','per-shardcompletionfilename'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};out=R/'execution_sources';out.mkdir(exist_ok=True);(out/f'{stage}_rank{rank}.py').write_text(actual);(out/f'{stage}_rank{rank}_receipt.json').write_text(json.dumps(receipt,indent=2));print('SHARD_START',stage,rank,'pid',os.getpid(),flush=True);sys.argv=[str(source),phase];exec(compile(modified,str(source),'exec'),{'__name__':'__main__','__file__':str(source),'select_shard':select_shard})
