"""Execute the complete delivered notebook using its named kernel and saved actual data."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nisar_water.common import write_json,sha256

if __name__=="__main__":
    root=Path(__file__).resolve().parents[1]
    path=root/"notebooks/NISAR_Water_Exploration.ipynb"
    notebook=nbformat.read(path,as_version=4)
    client=NotebookClient(notebook,timeout=240,kernel_name="nisar-water",allow_errors=False,
                          resources={"metadata":{"path":str(root)}})
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook,path)
    code=[cell for cell in notebook.cells if cell.cell_type=="code"]
    errors=[output for cell in code for output in cell.outputs if output.output_type=="error"]
    report={"state":"passed" if not errors else "failed","code_cells":len(code),
            "executed_code_cells":sum(cell.execution_count is not None for cell in code),
            "saved_outputs":sum(len(cell.outputs) for cell in code),"errors":len(errors),
            "notebook_sha256":sha256(path),"kernel":"nisar-water","mode":"cached_actual_data_with_fresh_selected_inference"}
    write_json(root/"evidence/notebook-execution.json",report)
    print(report)

