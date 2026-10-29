"""Prepare or run pinned official OPERA NISAR stages on existing development crops."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json
from nisar_water.opera_adapter import prepare_job,run_official_worker,compare_job


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='operation',required=True)
    p=sub.add_parser('prepare');p.add_argument('--case',required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--job',type=Path,required=True)
    p=sub.add_parser('worker');p.add_argument('request',type=Path)
    p=sub.add_parser('score');p.add_argument('request',type=Path);p.add_argument('--support',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.operation=='prepare':
        e=read_json(ROOT/'data/new-images-20261028'/args.case/f'acquisition-{args.index}.json')['observation']
        request=prepare_job(e,args.job,ROOT/'vendor/DSWX-SAR')
        print('Prepared official OPERA job:',request['job'])
    elif args.operation=='worker':run_official_worker(args.request.resolve())
    else:print(compare_job(args.request,args.support,args.output))

