"""Notebook helpers for inspecting actual acquisition, model and benchmark artifacts."""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import xarray as xr
from shapely.geometry import shape
from .common import read_json
from .inputs import load_catalog_observation


def entries_for_case(data_root, case_id):
    entries=[]
    for path in sorted((Path(data_root)/case_id).glob("acquisition-*.json")):
        record=read_json(path)
        if record.get("state") in ("acquired","cached"):
            entries.append(record["observation"])
    return sorted(entries,key=lambda e:e["acquired_at"])


def metrics_table(report):
    rows=[]
    for row in report["rows"]:
        for name,model in row.get("models",{}).items():
            metric=model.get("metrics") or {}
            rows.append({"location":row["case"],"radar_utc":row["datetime"],"model":name,
                         "reference":row["reference_state"],"evaluated_ha":metric.get("evaluated_ha"),
                         "precision":metric.get("precision"),"recall":metric.get("recall"),"IoU":metric.get("iou"),
                         "false_water_ha":metric.get("false_water_ha"),"missed_water_ha":metric.get("missed_water_ha")})
    return pd.DataFrame(rows)


def asset_path(series_root, observation, name):
    return Path(series_root)/observation["assets"][name]["url"].removeprefix("/assets/")


def plot_series(series_root):
    import matplotlib.pyplot as plt
    from PIL import Image
    series=read_json(Path(series_root)/"series.json")
    count=len(series["observations"])
    fig,axes=plt.subplots(2,count,figsize=(6*count,10),squeeze=False,constrained_layout=True)
    for i,observation in enumerate(series["observations"]):
        for row,key in enumerate(("sar","composite")):
            with Image.open(asset_path(series_root,observation,key)) as image:
                axes[row,i].imshow(image)
            axes[row,i].set_title(observation["datetime"][:10]+(" | HH input" if row==0 else " | predicted water"))
            axes[row,i].axis("off")
    fig.suptitle(series["id"]+" — fixed display range, frozen spatial-v2 model",fontsize=15)
    return fig


def plot_benchmark_overview(results_root):
    import matplotlib.pyplot as plt
    from PIL import Image
    results_root=Path(results_root)
    report=read_json(results_root/"report.json")
    records={r["observation_id"]:r for r in report["rows"] if r.get("observation_id")}
    roots=sorted((results_root/"series").glob("*/series.json"))
    fig,axes=plt.subplots(len(roots),2,figsize=(13,5.3*len(roots)),squeeze=False,constrained_layout=True)
    for row,path in enumerate(roots):
        series=read_json(path)
        for col,observation in enumerate(series["observations"][:2]):
            with Image.open(asset_path(path.parent,observation,"composite")) as image:
                axes[row,col].imshow(image)
            metric=records[observation["id"]]["models"]["spatial_v2"]["metrics"]
            if metric:
                precision='undefined' if metric['precision'] is None else f"{metric['precision']:.1%}"
                recall='undefined' if metric['recall'] is None else f"{metric['recall']:.1%}"
                status=f"P={precision} | R={recall} (weak reference)"
            else:
                status="Unscored: no retained dated reference"
            axes[row,col].set_title(series['id']+' | '+observation['datetime'][:10]+'\n'+status,fontsize=11)
            axes[row,col].axis("off")
    fig.suptitle("All eight new NISAR observations — frozen model, including failures",fontsize=16)
    return fig


def plot_errors(data_root, results_root, case_id, observation_id):
    import matplotlib.pyplot as plt
    data_root,results_root=Path(data_root),Path(results_root)
    entry=next(e for e in entries_for_case(data_root,case_id) if e["id"]==observation_id)
    o=load_catalog_observation(entry)
    folder=results_root/"measurements"/case_id/observation_id
    ref=read_json(data_root/case_id/"references"/observation_id/"reference.json")
    if not (folder/"reference-support.npz").exists():
        return None,"No optical acquisition within the declared 24-hour reference window."
    with np.load(folder/"reference-support.npz") as archive:
        raw,classes,evaluate=[archive[k] for k in ("raw_scl","classes","evaluate")]
    if not evaluate.any():
        return None,"The dated optical reference has no retained clear SCL4/5/6 interior cells; no accuracy is scored."
    rgb=np.zeros((3,*o.valid.shape),np.uint8)
    with rasterio.open(ref["local_assets"]["visual"]["path"]) as source:
        for band in range(3):
            reproject(source.read(band+1),rgb[band],src_transform=source.transform,src_crs=source.crs,
                      dst_transform=o.transform,dst_crs=o.crs,resampling=Resampling.bilinear)
    rgb=np.moveaxis(rgb,0,-1)
    rgb[~np.isin(raw,[4,5,6])]=[40,40,40]
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    axes[0].imshow(rgb);axes[0].set_title("Dated optical context | cloud/unknown masked")
    for axis,name in zip(axes[1:],("previous_v1","spatial_v2")):
        with np.load(folder/(name+".npz")) as archive:
            labels=archive["labels"]
        colors=np.zeros((*o.valid.shape,3),np.uint8)
        colors[evaluate & (classes!=6) & (labels==0)]=[90,100,110]
        colors[evaluate & (classes==6) & (labels==1)]=[35,153,245]
        colors[evaluate & (classes==6) & (labels==0)]=[255,160,44]
        colors[evaluate & (classes!=6) & (labels==1)]=[235,66,194]
        axis.imshow(colors);axis.set_title(name+" | reference errors")
    for axis in axes:axis.axis("off")
    return fig,"Blue: correct water; orange: missed water; magenta: false water; gray: correct land; black: unscored."


def to_notebook_cube(entries):
    observations=[load_catalog_observation(e) for e in entries]
    if not observations:
        raise ValueError("No observations selected")
    first=observations[0]
    if any((o.valid.shape,o.transform,o.crs)!=(first.valid.shape,first.transform,first.crs) for o in observations):
        raise ValueError("Acquisitions use different native analysis grids. Keep them separate; the time-series pipeline handles alignment explicitly.")
    arrays=np.stack([np.where(o.valid[None],o.power,np.nan) for o in observations])
    h,w=first.valid.shape;t=first.transform
    return xr.Dataset({c:(("time","y","x"),arrays[:,i]) for i,c in enumerate(("HHHH","HVHV"))},
        coords={"time":np.array([o.metadata["acquired_at"].removesuffix("Z") for o in observations],dtype="datetime64[s]"),
                "nombre":("time",[e["granule_id"]+".h5" for e in entries]),
                "x":t.c+(np.arange(w)+.5)*t.a,"y":t.f+(np.arange(h)+.5)*t.e},
        attrs={"epsg":first.crs.to_epsg(),"frecuencia":"A","multilook":1,"units":"gamma0_power",
               "descripcion":"gamma0 (RTC) en potencia lineal, NISAR L2 GCOV; 20m prepared grid",
               "aoi_wkt":shape(entries[0]["aoi"]).wkt})


def plot_water_features(data_root, results_root, experiment):
    import matplotlib.pyplot as plt
    with np.load(Path(experiment)/"development/train.npz") as archive:
        original=archive["context"][archive["y"]==1,:2]
    distributions={"Training water":original}
    for case in read_json(Path(data_root)/"plan.json")["cases"]:
        for entry in entries_for_case(data_root,case["id"]):
            folder=Path(results_root)/"measurements"/case["id"]/entry["id"]
            path=folder/"reference-support.npz"
            if not path.exists():continue
            with np.load(path) as archive:
                water=(archive["classes"]==6)&archive["evaluate"]
            if not water.any():continue
            o=load_catalog_observation(entry)
            distributions[case["id"]+" "+entry["acquired_at"][:10]]=10*np.log10(o.power[:,water]).T
    rows=[]
    fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    for i,axis in enumerate(axes):
        values=[array[:,i] for array in distributions.values()]
        axis.boxplot(values,orientation="horizontal",tick_labels=list(distributions),showfliers=False)
        axis.set_xlabel(("HH" if i==0 else "HV")+" gamma0 (dB)")
        axis.grid(axis="x",alpha=.2)
    for name,values in distributions.items():
        rows.append({"support":name,"HH_median_db":float(np.median(values[:,0])),"HV_median_db":float(np.median(values[:,1])),"cells":len(values)})
    return fig,pd.DataFrame(rows)

