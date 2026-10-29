"""Check notebook outputs, credential absence, new-data hashes and frozen model."""
from pathlib import Path
import re
import nbformat
from nisar_water.common import read_json,write_json,sha256


if __name__=="__main__":
    root=Path(__file__).resolve().parents[1]
    notebook=root/'notebooks/NISAR_Water_Exploration.ipynb'
    nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
    jwt=re.compile(r'eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{40,}')
    scanned=[]
    for directory in ['src','scripts','notebooks','data/new-images-20261028','outputs/new-images-20261028','evidence']:
        for path in (root/directory).rglob('*'):
            if path.is_file() and path.suffix in {'.py','.ipynb','.json','.md','.csv','.txt'}:
                assert not jwt.search(path.read_text(encoding='utf-8',errors='replace')),str(path)
                scanned.append(path)
    plan=read_json(root/'data/new-images-20261028/plan.json')
    frozen=read_json(root/'models/spatial-v2/freeze.json')
    assert sha256(root/'models/spatial-v2/selected/model.json')==frozen['model_sha256']==plan['model_sha256']
    assert sha256(root/'models/spatial-v2/selected/estimator.joblib')==frozen['estimator_sha256']
    images=0;radar_bytes=0
    for path in (root/'data/new-images-20261028').glob('*/acquisition-*.json'):
        receipt=read_json(path)
        assert receipt['state'] in {'acquired','cached'}
        images+=1;radar_bytes+=receipt.get('transfer_bytes',0)
        for asset in receipt['observation']['assets'].values():
            assert sha256(asset['reference'])==asset['sha256']
    assert images==8
    report=read_json(root/'outputs/new-images-20261028/report.json')
    assert len(report['rows'])==8 and report['model_unchanged']
    result={'state':'passed','notebook_sha256':sha256(notebook),'executed_code_cells':sum(c.cell_type=='code' for c in nb.cells),
            'new_radar_images':images,'new_regions':len(plan['cases']),'scored_images':report['aggregate']['spatial_v2']['scored_images'],
            'radar_range_bytes':radar_bytes,'credential_scan_files':len(scanned),'plaintext_jwt_found':False,'checkpoint_unchanged':True,
            'model_assessment':read_json(root/'outputs/new-images-20261028/assessment.json')['state']}
    write_json(root/'evidence/notebook-delivery-verification.json',result)
    print(result)

