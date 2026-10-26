"""Compare a CPU rebuild with the closed server MC reference snapshot."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from common import ROOT,save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference-dir',default='audit_reference/mc');args=parser.parse_args()
    reference=ROOT/args.reference_dir
    metadata=json.loads((reference/'REFERENCE_PROVENANCE.json').read_text())
    assert metadata['passed'] and metadata['validation_configurations_preserved'] and metadata['raw_mc_files_unchanged']==18
    for name,digest in metadata['analysis_source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    assert len(metadata['raw_mc_file_manifest'])==18
    for name,entry in metadata['raw_mc_file_manifest'].items():
        payload=(ROOT/name).read_bytes()
        assert len(payload)==entry['bytes'] and hashlib.sha256(payload).hexdigest()==entry['sha256'],name
    tables=[]
    for source in sorted((reference/'results').glob('*.csv')):
        expected=pd.read_csv(source);actual=pd.read_csv(ROOT/'results'/source.name)
        assert len(expected)==len(actual),(source.name,len(expected),len(actual))
        assert set(expected.columns)==set(actual.columns),source.name
        for column in expected.columns:
            a=expected[column];b=actual[column]
            assert np.array_equal(a.isna(),b.isna()),(source.name,column,'missingness')
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                assert np.allclose(a.to_numpy(float),b.to_numpy(float),rtol=1e-12,atol=1e-12,equal_nan=True),(source.name,column)
            else:
                assert a.fillna('').astype(str).tolist()==b.fillna('').astype(str).tolist(),(source.name,column)
        tables.append(dict(file=source.name,rows=len(expected),shared_columns=len(expected.columns),
                           all_columns_equal=True))
    assert len(tables)==15
    for source in sorted((reference/'configs').glob('*.json')):
        expected=json.loads(source.read_text());actual=json.loads((ROOT/'configs'/source.name).read_text())
        assert actual==expected,source.name
    result=dict(passed=True,tables=tables,configurations_exactly_equal=['mc_selected.json','halting_selected.json'],
                numeric_relative_tolerance=1e-12,numeric_absolute_tolerance=1e-12,
                analysis_source_hashes_match=True,analysis_source_sha256=metadata['analysis_source_sha256'],
                raw_mc_snapshot_sha256=metadata['raw_snapshot_archive_sha256'],
                reference_archive_sha256=metadata['archive_sha256'],
                scope='Closed MC-only CPU rebuild: all reference CSV columns/values and exact validation-selected configurations under identical analysis source. Mathematical generation, real stopping and final delivery are excluded.')
    save('results/mc_rebuild_reference_check.json',result)
    print(json.dumps(dict(passed=True,tables=len(tables),rows=sum(r['rows'] for r in tables),scope='MC reference fields only; final full rebuild still pending')),flush=True)


if __name__=='__main__':main()
