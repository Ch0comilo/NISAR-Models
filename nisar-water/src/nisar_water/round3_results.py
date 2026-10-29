"""Saved evidence, common-grid comparison figures and readiness assessment."""
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from .common import read_json,write_json,sha256,utcnow
from .round3_evaluation import load_scene,aggregate
from .review import aligned_rgb

ROOT=Path(__file__).resolve().parents[2]


def cohort_specs(name):
    if name=='test':return [s for s in read_json(ROOT/'data/round3-prepared/manifest.json')['scenes'] if s['split']=='test']
    dataset={'challenge':'round3-challenge','shorelines':'round3-shorelines'}[name]
    return read_json(ROOT/'data'/dataset/'manifest.json')['scenes']


def cohort_table(report,cohort):
    rows=[]
    for family,results in report['families'].items():
        for region,row in results['regions'].items():
            rows.append({'cohort':cohort,'family':family,'region':region,'scenes':row['scenes'],
                         **{k:row[k] for k in ('precision','recall','false_positive_rate','iou','reference_water_ha','reference_nonwater_ha','reference_coverage')}})
    return rows


def native_composite(o,labels):
    db=10*np.log10(np.where(o.valid,o.power[0],1));gray=np.uint8(np.clip((db+30)/33,0,1)*255)
    rgb=np.repeat(gray[...,None],3,axis=2);rgb[~o.valid|(o.weights<=0)]=[235,237,239]
    water=labels==1;rgb[water]=np.uint8(.32*rgb[water]+.68*np.array([35,153,245]))
    border=water&~binary_erosion(water,structure=np.ones((3,3)));rgb[border]=[126,233,255]
    return rgb


def plot_cohort(name,*,comparison=True):
    import matplotlib.pyplot as plt
    specs=cohort_specs(name);columns=5 if comparison else 3
    fig,axes=plt.subplots(len(specs),columns,figsize=(3.4*columns,3.4*len(specs)),squeeze=False,layout='constrained')
    for row,spec in enumerate(specs):
        o,truth,evaluate,sidecar=load_scene(spec);rgb=aligned_rgb(o,sidecar)
        db=10*np.log10(np.where(o.valid,o.power[0],1));db[~o.valid|(o.weights<=0)]=np.nan
        axes[row,0].imshow(db,cmap='gray',vmin=-30,vmax=3);axes[row,0].set_title(spec['region']+' | '+spec['acquired_at'][:10]+'\nHH gamma0 [-30,3] dB',fontsize=9)
        axes[row,1].imshow(rgb);axes[row,1].set_title('Dated Sentinel-2 context',fontsize=10)
        families=['hgb','unet','opera'] if comparison else ['hgb']
        for col,family in enumerate(families,start=2):
            with np.load(ROOT/'outputs/round3'/(name+'-predictions')/family/spec['id']/'prediction.npz',allow_pickle=False) as z:
                labels=z['labels']
            axes[row,col].imshow(native_composite(o,labels));axes[row,col].set_title(family.upper()+' | frozen decision',fontsize=10)
        for ax in axes[row]:ax.set_axis_off()
    fig.suptitle({'test':'Original held-out lake crops — mostly open-water interiors',
                  'challenge':'Hard-negative challenge — airport and mountainous land',
                  'shorelines':'Additional prospective shorelines — Powell and Malawi'}[name]+'\nAll scenes retained; blue is predicted open water, not verified truth',fontsize=13)
    return fig


def plot_errors(name,scene_id,family='hgb'):
    import matplotlib.pyplot as plt
    spec=next(s for s in cohort_specs(name) if s['id']==scene_id);o,t,e,side=load_scene(spec)
    with np.load(ROOT/'outputs/round3'/(name+'-predictions')/family/spec['id']/'prediction.npz',allow_pickle=False) as z:l=z['labels']
    reference=np.full((*t.shape,3),232,np.uint8);reference[e&(t==0)]=[167,187,163];reference[e&(t==1)]=[35,153,245]
    errors=np.full_like(reference,240);errors[e&(t==1)&(l==1)]=[35,153,245]
    errors[e&(t==0)&(l==1)]=[217,76,61];errors[e&(t==1)&(l==0)]=[244,183,55]
    fig,axes=plt.subplots(1,4,figsize=(16,4.4),layout='constrained')
    for ax,data,title in zip(axes,[aligned_rgb(o,side),reference,native_composite(o,l),errors],
        ['Optical context','Weak interiors: blue water, green land',family.upper()+' predicted water','Red false water; gold missed water; blue correct water']):
        ax.imshow(data);ax.set_title(title,fontsize=9);ax.axis('off')
    fig.suptitle(spec['region']+' '+spec['acquired_at'][:10]+' — unknown/boundary pixels are unscored')
    return fig


def build_assessment():
    out=ROOT/'outputs/round3';freeze=read_json(out/'freeze.json')
    cohorts={name:read_json(out/(name+'-report.json')) for name in ('test','challenge','shorelines')}
    rows=[row for name,report in cohorts.items() for row in cohort_table(report,name)]
    pooled={}
    for family in ('hgb','unet','opera'):
        scenes=[s for report in cohorts.values() for s in report['families'][family]['scenes']]
        pooled[family]=aggregate(scenes)
    unchanged=sha256(ROOT/'models/round3/selected/model.json')==freeze['selected_model_sha256']
    unchanged&=all(sha256(Path(s['directory'])/'model.json')==s['model_sha256'] for s in freeze['families'].values())
    unchanged&=all(sha256(ROOT/p)==h for p,h in freeze['implementation_hashes'].items())
    write_json(out/'report.json',{'created_at':utcnow(),'selected_family':freeze['selected_family'],'model_unchanged':bool(unchanged),
        'cohorts':cohorts,'region_table':rows,'pooled_across_cohorts':pooled,'new_scored_images':13,'new_scored_regions':8,
        'selection_after_test':False,'label_kind':'dated_SCL_spectral_consensus_interiors',
        'limitations':['Original lake test is highly water-dominated; additional cohorts are reported separately',
                       'Area-weighted pooled scores are not an estimate of global accuracy',
                       'No independent expert shoreline truth; weak boundary diagnostic is not shoreline accuracy']})
    assessment={'state':'not_ready_for_general_use','created_at':utcnow(),'selected_model':freeze['selected_family'],
        'model_sha256':freeze['selected_model_sha256'],'model_unchanged_after_test':bool(unchanged),
        'scope':'Headless research prototype, current frequency-A HH/HV NISAR L-GCOV UTM crops',
        'evidence':{'train_scenes':28,'validation_scenes':5,'original_test_scenes':7,'additional_stress_scenes':2,
                    'additional_shoreline_scenes':4,'independent_expert_reviews':0},
        'blocking_findings':[
            {'region':'denver-airport','finding':'Both learned models mistake extensive clear land for water',
             'hgb_land_false_positive_rate':cohorts['challenge']['families']['hgb']['regions']['denver-airport']['false_positive_rate'],
             'unet_land_false_positive_rate':cohorts['challenge']['families']['unet']['regions']['denver-airport']['false_positive_rate']},
            {'region':'powell-shore','finding':'Dry shoreline terrain remains difficult for all three approaches',
             'precision':{f:cohorts['shorelines']['families'][f]['regions']['powell-shore']['precision'] for f in pooled}},
            {'region':'constance','finding':'Official-algorithm crop adapter missed all-water validation crops; full-frame OPERA equivalence unproven'},
            {'finding':'Weak optical interiors cannot establish independent shoreline or global accuracy'}],
        'usable_now':['Actual NISAR inference without an application UI','Blue water and border PNGs, COG masks, GeoJSON and JSON API',
                      'Every compatible observation in a bounded AOI/date query, with partial runs explicit',
                      'Same-geometry temporal comparisons on common classified support','Reproducible frozen-family comparison and blinded annotation packets'],
        'next_scientific_work':['Expert label representative shorelines, dry soil, airports, mountains and flooded vegetation',
            'Add new hard-negative geographies to training and validation; reserve another untouched geographic test',
            'Evaluate physical context such as correctly interpolated incidence, DEM slope/HAND, and causal past-date persistence',
            'Treat historical water as a soft prior with explicit emerging-water tests; do not erase new water with a hard old-water mask',
            'Calibrate or abstain using independent validation; do not interpret raw scores as probabilities',
            'Recheck generalization before deploying another model; a larger architecture alone did not resolve ambiguity']}
    write_json(out/'assessment.json',assessment);return assessment
