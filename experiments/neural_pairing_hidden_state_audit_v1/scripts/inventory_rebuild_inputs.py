"""Inventory original input recipe without downloading corpora or using GPU."""
import json
import runpy
from linux_runtime import ROOT,configure,save

configure('cpu')
module=runpy.run_path(str(ROOT/'upstream/pairing/build_coread.py'),run_name='recipe_inventory')
specs=module['PASSAGES']+module['TASKS']
sources=[dict(name=s[0],repo=s[1],config=s[2],split=s[3],weight=s[5] if len(s)==8 else s[4],
    category='passage' if len(s)==8 else 'task') for s in specs]
manifest=dict(status='recipe_inventory_not_rebuilt',sources=sources,source_count=len(sources),
    default_max_train_items_per_source=5000,default_max_test_items_per_source=100,
    source_holdout='Every40th source example/group; final content dedup against training',
    missing_original_generated_files=['coread.json','talk.json','play.json','replies.json with steering and distillation targets'],
    completed_inputs=['pools.json','byte-identical16631/595/3165 QA bench.json','620 regenerated own replies'],
    generation_defaults=dict(steering_candidates_per_model=4000,steering_kept_target_per_model=2500,
        distillation_train_items=6000,distillation_test_items=600,distillation_max_tokens=128,
        play_train_target=1600,play_test_target=300,play_turns=6),
    caveat='Builder defaults and public scripts are available. Original successful source set, source revisions and generated-target files are not supplied; a regenerated mixture must be labeled as such.')
save(ROOT/'configs/REBUILD_INPUT_INVENTORY.json',manifest)
print(json.dumps(dict(source_count=len(sources),unique_dataset_repos=len({s['repo'] for s in sources}),status=manifest['status'])))
