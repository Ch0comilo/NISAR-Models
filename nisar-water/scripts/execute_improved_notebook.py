from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nisar_water.common import write_json,sha256
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];path=root/'notebooks/NISAR_Water_Improved_Classifier.ipynb'
    notebook=nbformat.read(path,as_version=4)
    NotebookClient(notebook,timeout=300,kernel_name='nisar-water',allow_errors=False,resources={'metadata':{'path':str(root)}}).execute()
    nbformat.validate(notebook);nbformat.write(notebook,path)
    cells=[c for c in notebook.cells if c.cell_type=='code'];errors=[o for c in cells for o in c.outputs if o.output_type=='error']
    result={'state':'passed' if not errors else 'failed','code_cells':len(cells),'executed_code_cells':sum(c.execution_count is not None for c in cells),
        'saved_outputs':sum(len(c.outputs) for c in cells),'errors':len(errors),'notebook_sha256':sha256(path),'kernel':'nisar-water',
        'mode':'cached_actual_data_fresh_inference_pila_HTTP_API','default_satellite_downloads':False}
    write_json(root/'evidence/improved-notebook-execution.json',result);print(result)
