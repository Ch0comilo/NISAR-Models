"""Development-only weak-label QA; new reserved test imagery stays undisplayed."""
from pathlib import Path
import sys
import math
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nisar_water.common import read_json,write_json,sha256,utcnow
if __name__=='__main__':
    manifest=read_json(ROOT/'data/round4-prepared/scenes-manifest.json')
    specs=[s for s in manifest['scenes'] if s['split']!='test' and 'previous_split' not in s]
    width,height=620,330;columns=3;rows=math.ceil(len(specs)/columns)
    canvas=Image.new('RGB',(width*columns,height*rows),'#17232f');records=[]
    for i,spec in enumerate(specs):
        path=Path(spec['path']);metadata=read_json(path.with_suffix('.json'))
        image=Image.open(path.with_suffix('.qa.png')).convert('RGB');image.thumbnail((width-10,height-40))
        x=(i%columns)*width;y=(i//columns)*height
        canvas.paste(image,(x+(width-image.width)//2,y+35))
        ImageDraw.Draw(canvas).text((x+8,y+6),spec['region']+' '+spec['acquired_at'][:10]+' '+spec['split'],fill='white')
        records.append({'id':spec['id'],'region':spec['region'],'split':spec['split'],'counts':metadata['counts'],
                        'qa_image_sha256':sha256(path.with_suffix('.qa.png'))})
    destination=ROOT/'evidence/round4-development-label-qa.png';canvas.save(destination)
    write_json(ROOT/'evidence/round4-development-qa-pending.json',{'created_at':utcnow(),'records':records,
        'montage_sha256':sha256(destination),'review_status':'pending_assistant_visual_inspection','expert_reviewed':False,
        'new_test_images_included':False,'model_scores_used':False})
    print(len(specs),'new development scenes; no reserved test images included')
