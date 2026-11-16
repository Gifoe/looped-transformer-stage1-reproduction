import pathlib,json,subprocess,datetime
R=pathlib.Path('/root/ouro-binding-interface-sol-20261117');P='/root/autodl-tmp/ouro-e4-e5/venv/bin/python'
subprocess.run([P,'-u',str(R/'code/technical.py')],cwd=R,check=True);subprocess.run([P,'-u',str(R/'code/oracle.py')],cwd=R,check=True);children=[];(R/'logs').mkdir(exist_ok=True)
for rank in range(3):
 stream=(R/'logs'/f'discovery_rank{rank}.log').open('w');child=subprocess.Popen([P,'-u',str(R/'code/run_discovery.py'),str(rank),'3'],cwd=R,stdout=stream,stderr=subprocess.STDOUT);children.append((child,stream));print('DISCOVERY_SHARD_LIVE',rank,child.pid,flush=True)
for child,stream in children:code=child.wait();stream.close();assert code==0,('discoveryshardfailed',child.pid,code)
subprocess.run([P,'-u',str(R/'code/select_layer.py')],cwd=R,check=True);print('DEVELOPMENT_EVENT_DISCOVERY_COMPLETE',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
