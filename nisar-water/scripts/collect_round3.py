from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from nisar_water.common import write_json
from nisar_water.round3_data import create_plan,collect_site


if __name__=="__main__":
    root=Path("data/round3").resolve();plan=create_plan(root)
    reference=Path("../nisar-landcover-phase1").resolve()
    results=[]
    with ProcessPoolExecutor(max_workers=2) as executor:
        jobs={executor.submit(collect_site,site,root,reference):site["id"] for site in plan["sites"]}
        for future in as_completed(jobs):
            try:results.append(future.result())
            except Exception as error:results.append({"site_id":jobs[future],"state":"failed","error":type(error).__name__,"message":str(error)[:500]})
            write_json(root/"collection-summary.json",{"results":results,"fitting_performed":False,"test_predictions_inspected":False})

