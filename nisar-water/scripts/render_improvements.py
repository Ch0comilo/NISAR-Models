from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nisar_water.improvement_results import build_assessment,plot_atlas,plot_error
from nisar_water.common import read_json
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('round',type=int,choices=[4,5]);a=p.parse_args();experiment=a.round
    build_assessment(experiment)
    for brief in (False,True):
        fig=plot_atlas(experiment,one_per_region=brief);suffix='overview' if brief else 'all-test-images'
        fig.savefig(ROOT/f'evidence/round{experiment}-{suffix}.png',dpi=120);plt.close(fig)
    regions=read_json(ROOT/f'outputs/round{experiment}/test-report.json')['families']['radar']['regions']
    for region in regions:
        fig=plot_error(experiment,region);fig.savefig(ROOT/f'evidence/round{experiment}-{region}-errors.png',dpi=130);plt.close(fig)
    print('Saved round',experiment,'assessment, all-image atlas and region error figures')
