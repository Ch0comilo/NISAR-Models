from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json
from nisar_water.review import create_review_pack
if __name__=='__main__':
    specs=[s for s in read_json(ROOT/'data/round3-prepared/manifest.json')['scenes'] if s['split']=='test']
    for dataset in ('round3-challenge','round3-shorelines'):specs+=read_json(ROOT/'data'/dataset/'manifest.json')['scenes']
    index=create_review_pack(specs,ROOT/'review/round3');print(len(index),'blinded packets; none human-reviewed')
