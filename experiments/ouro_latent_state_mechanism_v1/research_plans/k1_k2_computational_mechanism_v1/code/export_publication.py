import pathlib,json,sys,base64,hashlib
P=pathlib.Path('/root/ouro-computation-sol-20261116/publication');manifest=json.loads((P/'PUBLICATION_MANIFEST.json').read_text());files=manifest['files'];d=(P/'PUBLICATION_MANIFEST.json').read_bytes();files.append({'path':'PUBLICATION_MANIFEST.json','bytes':len(d),'sha256':hashlib.sha256(d).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest()});action=sys.argv[1]
if action=='index':print(json.dumps({'files':files,'total_bytes':sum(x['bytes'] for x in files)}))
elif action=='small':
 offset=int(sys.argv[2]);eligible=[x for x in files if x['bytes']<=50000 and pathlib.Path(x['path']).suffix in ('.py','.md','.json','.jsonl','.svg')];batch=[];size=0
 for x in eligible[offset:]:
  if size+x['bytes']>100000 and batch:break
  batch.append({**x,'content':(P/x['path']).read_bytes().decode('utf-8')});size+=x['bytes']
 print(json.dumps({'files':batch,'next':offset+len(batch),'total':len(eligible)}))
elif action=='chunk':
 name=sys.argv[2];allowed={x['path'] for x in files};assert name in allowed;p=(P/name).resolve();assert P.resolve() in p.parents;offset=int(sys.argv[3]);length=int(sys.argv[4]);assert 0<length<=737280
 with p.open('rb') as f:f.seek(offset);d=f.read(length)
 print(json.dumps({'path':name,'offset':offset,'bytes':len(d),'base64':base64.b64encode(d).decode()}))
