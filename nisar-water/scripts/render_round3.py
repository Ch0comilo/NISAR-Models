from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nisar_water.round3_results import plot_cohort,plot_errors,cohort_specs,build_assessment
if __name__=='__main__':
    build_assessment()
    for name in ('test','shorelines','challenge'):
        figure=plot_cohort(name,comparison=name!='test');figure.savefig(ROOT/'evidence'/('round3-'+name+'-comparison.png'),dpi=130);plt.close(figure)
    for name,region in [('shorelines','powell-shore'),('shorelines','malawi-shore'),('challenge','denver-airport')]:
        spec=next(s for s in cohort_specs(name) if s['region']==region)
        figure=plot_errors(name,spec['id']);figure.savefig(ROOT/'evidence'/('round3-'+region+'-errors.png'),dpi=150);plt.close(figure)
    print('Saved three full atlases, three error figures, report and readiness assessment')
