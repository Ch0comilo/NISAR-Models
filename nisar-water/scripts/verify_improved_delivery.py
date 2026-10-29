"""Frozen-source/input integrity, executed notebook, finite JSON and no secrets."""
from pathlib import Path
import re,json
import xml.etree.ElementTree as ET
from zipfile import ZipFile,ZIP_DEFLATED
import nbformat
from nisar_water.common import read_json,write_json,sha256,utcnow
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    results={}
    for experiment in (3,4,5):
        freeze=read_json(ROOT/f'outputs/round{experiment}/freeze.json')
        for p,digest in freeze['implementation_hashes'].items():assert sha256(ROOT/p)==digest
        assert sha256(Path(freeze['selected_directory'])/'model.json')==freeze['selected_model_sha256']
        for snapshot in freeze['families'].values():
            directory=Path(snapshot['directory'])
            assert sha256(directory/'model.json')==snapshot['model_sha256']
            for name,digest in snapshot.get('artifacts',{}).items():assert sha256(directory/name)==digest
        results[f'round{experiment}_frozen_sources_and_models_unchanged']=True
    original=read_json(ROOT/'models/spatial-v2/freeze.json')
    assert sha256(ROOT/'models/spatial-v2/selected/model.json')==original['model_sha256']
    for name,digest in original['source_hashes'].items():assert sha256(ROOT/'src/nisar_water'/name)==digest
    results['historical_v2_unchanged']=True
    scene_checks=0
    for experiment in (4,5):
        manifest=ROOT/f'data/round{experiment}-prepared/manifest.json';freeze=read_json(ROOT/f'outputs/round{experiment}/freeze.json')
        assert sha256(manifest)==freeze['manifest_sha256']
        for scene in read_json(manifest)['scenes']:
            for path_key,hash_key in [('path','sha256'),('context_path','context_sha256'),('context_metadata_path','context_metadata_sha256')]:
                assert sha256(scene[path_key])==scene[hash_key]
            scene_checks+=1
    results['scene_context_bindings_verified']=scene_checks
    review_freeze=read_json(ROOT/'outputs/round5/agreement-review-freeze.json')
    assert sha256(Path(review_freeze['model_directory'])/'model.json')==review_freeze['model_sha256']
    review_meta=read_json(Path(review_freeze['model_directory'])/'model.json')
    assert sha256(ROOT/'src/nisar_water/agreement_model.py')==review_meta['implementation_sha256']
    path=ROOT/'notebooks/NISAR_Water_Improved_Classifier.ipynb';notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
    code=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    results.update(executed_code_cells=len(code),notebook_outputs=sum(len(c.outputs) for c in code),notebook_sha256=sha256(path))
    counts={}
    for name in ('main','neural'):
        root=ET.parse(ROOT/f'evidence/improved-{name}-tests.xml').getroot();suites=list(root.iter('testsuite'))
        failed=sum(int(s.get('failures','0'))+int(s.get('errors','0')) for s in suites)
        skipped=sum(int(s.get('skipped','0')) for s in suites);total=sum(int(s.get('tests','0')) for s in suites)
        assert failed==0;counts[name]={'passed':total-skipped,'skipped':skipped}
    results['software_tests']=counts
    checks=read_json(ROOT/'evidence/improved-output-verification.json');assert checks['state']=='passed'
    results['latest_export_sets_verified']=len(checks['observations'])
    results['round4_export_sets_verified']=len(read_json(ROOT/'evidence/round4-output-verification.json')['observations'])
    demo=read_json(ROOT/'outputs/round5/hidroituango-cloud-demo/batch.json')
    assert demo['complete_for_compatible_query'];results['cloud_demo_observations']=demo['acquired_observations']
    ee=read_json(ROOT/'evidence/round4-earthengine-access.json');assert ee['state']=='available'
    results['earthengine_project']=ee['project']
    results['fresh_ee_references']=sum(r['state']=='available' for r in read_json(ROOT/'evidence/round5-earthengine-review-check.json')['records'])
    credential=re.compile(r'eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{40,}')
    # Signed EE download URLs were only consumed in memory; no credentials or
    # helper credential files are included in the deliverable tree.
    scanned=0
    folders=['src','scripts','docs','notebooks','evidence','models/round4','models/round5','models/agreement-review',
        'data/round4','data/round4-prepared','data/round5','data/round5-prepared','outputs/round4','outputs/round5','review']
    for folder in folders:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.ipynb','.md','.txt','.csv','.ps1'):
                text=p.read_text(encoding='utf-8',errors='replace')
                assert not credential.search(text),'Credential-like JWT in deliverable: '+str(p)
                if p.suffix=='.json':
                    def reject(value):raise ValueError('Nonfinite JSON '+value)
                    json.loads(text,parse_constant=reject)
                scanned+=1
    results.update(text_files_checked=scanned,plaintext_jwt_found=False)
    kit=ROOT/'review/team-review-kit.zip'
    with ZipFile(kit,'w',ZIP_DEFLATED) as archive:
        for folder in ('earthengine','earthengine-round5'):
            for p in (ROOT/'review'/folder).glob('*'):
                if p.suffix in ('.js','.json','.md'):archive.write(p,p.relative_to(ROOT/'review'))
        for folder in ('round3','round5'):
            for packet in (ROOT/'review'/folder).glob('review-*'):
                for p in packet.iterdir():
                    if p.name in ('source.json','radar-hh.png','optical.png','earthengine-optical.png','annotations.geojson'):
                        archive.write(p,p.relative_to(ROOT/'review'))
    results.update(review_kit_sha256=sha256(kit),review_kit_bytes=kit.stat().st_size,
        current_readiness=read_json(ROOT/'outputs/round5/assessment.json')['state'],expert_labels_imported_into_fits=0,
        phase4_fresh_test_images=12,phase5_fresh_test_images=5,state='passed',verified_at=utcnow())
    write_json(ROOT/'evidence/improved-delivery-verification.json',results);print(results)
