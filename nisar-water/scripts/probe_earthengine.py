"""Use existing credentials without displaying or replacing any credential."""
from pathlib import Path
import ee
from nisar_water.common import write_json,utcnow
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    result={'checked_at':utcnow(),'project':'nisac-512902','credential_contents_read_or_logged':False}
    stage='initialize'
    try:
        import httplib2
        ee.Initialize(project=result['project'],http_transport=httplib2.Http(timeout=30))
        ee.data.setDeadline(30000)
        stage='constant_request'
        value=ee.Number(1).getInfo()
        result['constant_response_type']=type(value).__name__
        assert value==1
        result.update(state='available',api_version=ee.__version__)
    except Exception as error:
        # Auth/client exceptions can contain request details; never dump them.
        import traceback
        result.update(state='unavailable',stage=stage,error_type=type(error).__name__,message='Existing credential/project request did not succeed',
            frames=[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(error.__traceback__)])
    write_json(ROOT/'evidence/round4-earthengine-access.json',result);print(result)
