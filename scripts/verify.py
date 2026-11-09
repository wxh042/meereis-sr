import subprocess,os,time,sys,json,io,secrets,tempfile
from pathlib import Path
import requests
from PIL import Image
root=Path(__file__).resolve().parents[1];base=root/'platform';runtime=sys.executable
(root/'runtime').mkdir(exist_ok=True)
tmp=tempfile.TemporaryDirectory(prefix='verify-',dir=root/'runtime',ignore_cleanup_errors=True);run=Path(tmp.name)
os.chdir(base)
env=os.environ.copy();env.update(SQLALCHEMY_DATABASE_URL='sqlite:///'+(run/'test.db').as_posix(),DEMO_PASSWORD=secrets.token_urlsafe(18),JWT_SECRET_KEY=secrets.token_urlsafe(48),CELERY_BROKER_URL='filesystem://',CELERY_RESULT_BACKEND=(run/'backend').as_uri()+'/')
env['CELERY_RESULT_BACKEND']='file://'+str(run/'backend').replace('\\','/')+'/'
(run/'backend').mkdir(parents=True,exist_ok=True)
env['MEEREIS_RUNTIME_DIR']=str(run)
env['API_BASE_URL']='http://127.0.0.1:18001'
procs=[];logs=[]
def start(args,name):
 f=open(run/f'{name}.log','w',encoding='utf-8');logs.append(f)
 p=subprocess.Popen([runtime,*args],cwd=base,env=env,stdout=f,stderr=f);procs.append(p)
 return p
def wait_http(url):
 for _ in range(90):
  try:
   r=requests.get(url,timeout=2)
   if r.status_code==200:return r
  except requests.RequestException:pass
  time.sleep(0.4)
 raise RuntimeError('Service not ready: '+url)
try:
 start(['-m','uvicorn','app:app','--host','127.0.0.1','--port','18001'],'api')
 wait_http('http://127.0.0.1:18001/')
 start(['-m','celery','-A','celery_worker.celery_app','worker','--pool=solo','--loglevel=info'],'worker')
 start(['-m','streamlit','run','web_gui.py','--server.port','18501','--server.headless','true'],'web')
 wait_http('http://127.0.0.1:18501/_stcore/health')
 url='http://127.0.0.1:18001'
 r=requests.post(url+'/login',data={'username':'researcher','password':env['DEMO_PASSWORD']});r.raise_for_status();headers={'Authorization':'Bearer '+r.json()['access_token']}
 def poll(task):
  for _ in range(120):
   r=requests.get(url+'/task_status/'+task,timeout=5);r.raise_for_status();j=r.json()
   if j['state']=='FAILURE':raise RuntimeError(j)
   if j['state']=='SUCCESS':return j
   time.sleep(.3)
  raise RuntimeError('Task timed out')
 results=[]
 for model in ['25km','10km']:
  with open(root/'examples/input.png','rb') as f:r=requests.post(url+'/predict',headers=headers,data={'model_type':model},files={'file':('input.png',f,'image/png')})
  assert r.ok, r.text; j=poll(r.json()['task_id']);output=Image.open(j['result']['result_path']);assert output.size==(128,128)
  results.append({'model':model,'state':j['state'],'output_size':list(output.size)})
 with open(root/'examples/input.png','rb') as f:data=f.read()
 r=requests.post(url+'/predict_batch',headers=headers,data={'model_type':'10km'},files=[('files',('one.png',data,'image/png')),('files',('two.png',data,'image/png'))]);r.raise_for_status();batch=poll(r.json()['task_id']);assert batch['result']['batch_size']==2
 assert requests.post(url+'/predict',files={'file':('x.png',data,'image/png')}).status_code==401
 assert requests.post(url+'/predict',headers=headers,data={'model_type':'invalid'},files={'file':('x.png',data,'image/png')}).status_code==400
 assert requests.post(url+'/predict',headers=headers,files={'file':('bad.png',b'not an image','image/png')}).status_code==400
 hist=requests.get(url+'/history').json();assert any(x['status']=='SUCCESS' for x in hist)
 report={'single_inference':results,'batch_size':2,'authentication':True,'invalid_model_rejected':True,'invalid_image_rejected':True,'history_persistence':True,'streamlit_health':True,'transport':'Celery filesystem worker + file result backend','database':'SQLite isolated smoke database','device':'CPU','full_training_repeated':False}
 (root/'docs/verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
finally:
 for p in reversed(procs):p.terminate()
 for p in procs:
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill()
 for f in logs:f.close()
 tmp.cleanup()



