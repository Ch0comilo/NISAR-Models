from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nisar_water.common import write_json,sha256
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];path=root/'notebooks/NISAR_Water_Round3_Comparison.ipynb'
    notebook=nbformat.read(path,as_version=4)
    client=NotebookClient(notebook,timeout=300,kernel_name='nisar-water',allow_errors=False,resources={'metadata':{'path':str(root)}})
    client.execute();nbformat.validate(notebook);nbformat.write(notebook,path)
    code=[c for c in notebook.cells if c.cell_type=='code'];errors=[o for c in code for o in c.outputs if o.output_type=='error']
    result={'state':'passed' if not errors else 'failed','code_cells':len(code),'executed_code_cells':sum(c.execution_count is not None for c in code),
        'saved_outputs':sum(len(c.outputs) for c in code),'errors':len(errors),'notebook_sha256':sha256(path),'kernel':'nisar-water',
        'mode':'cached_actual_data_fresh_inference_pila_exports_actual_HTTP_API','default_satellite_downloads':False}
    write_json(root/'evidence/round3-notebook-execution.json',result);print(result)
