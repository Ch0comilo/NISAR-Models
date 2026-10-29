from pathlib import Path
import sys,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json
from nisar_water.review import create_review_pack
if __name__=='__main__':
    specs=[s for s in read_json(ROOT/'data/round5-prepared/manifest.json')['scenes'] if s['split']=='test']
    create_review_pack(specs,ROOT/'review/round5')
    target=ROOT/'review/earthengine-round5';target.mkdir(parents=True,exist_ok=True)
    guide='''# Fresh held-out-region reference review

These five packets cover Yusufeli, Nam Theun1 and Alice Springs. They were
outside fitting and selection for round5. The original13 round3 packets have
since become exposed training regions; they are useful for label-quality work
but cannot establish new geographic generalization.

Open https://code.earthengine.google.com/?project=nisac-512902 and paste
`review_round5.js`. Choose CASE_INDEX, fill REVIEWER and run. The script shows
the exact dated harmonized Sentinel-2 optical reference and AOI, without model
predictions or historical-water layers. All five references were queried
successfully in the project. No extra EE image crop was downloaded for these
five packets; their existing local optical/radar previews remain available.

Select a layer ending in ::open_water, ::nonwater or ::exclude, then draw
polygons. Unlabelled space remains unknown. Exclude clouds, uncertain shoreline
and temporally ambiguous areas. Check the radar/optical timestamp difference.
Use the local packet's radar-hh.png or radar-power.tif for NISAR context.

The export button prepares an unstarted GeoJSON task. Review and start it
yourself in Tasks. Save/download annotations before switching case. No remote
export task was started by the assistant. Expert status is a reviewer
declaration and must not be set without a real qualified independent review.

Import returned polygons with:

    python -m nisar_water.cli import-review --packet review/round5/REVIEW_KEY/source.json --annotations REVIEWED.geojson --output review/imported

The import verifies scene identity, reviewer/date and geometry, preserves
unknowns, and versions the result. Frozen inputs/models/reports are unchanged.
No expert labels were imported into the current fits. Review labels support a
separate, explicitly versioned reference assessment or future training dataset.
'''
    (target/'README.md').write_text(guide,encoding='utf-8');print('Prepared5 fresh-region review packets; no model predictions included')
