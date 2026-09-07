#!/usr/bin/env python3
"""Check original PCM against the rendered lossless soundtrack, byte for byte."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
c=json.loads((ROOT/'project.json').read_text())
def pcm(path,start=0):
 return subprocess.check_output(['ffmpeg','-v','error','-ss',str(start),'-i',str(path),'-map','0:a:0','-t',str(c['frame_count']/c['fps']),'-f','s24le','-c:a','pcm_s24le','pipe:1'])
a=pcm(ROOT/c['reference'],c['start_frame']/c['fps'])
b=pcm(ROOT/'renders'/(c.get('output_name','recreated-first10')+'-lossless.mkv'))
r={'original_audio_reused':True,'duration_seconds':c['frame_count']/c['fps'],'lossless_master_audio_pcm_equal':a==b,'bytes_compared':len(a),'reference_pcm_sha256':hashlib.sha256(a).hexdigest(),'master_pcm_sha256':hashlib.sha256(b).hexdigest()}
p=ROOT/c.get('report_dir','reports');p.mkdir(parents=True,exist_ok=True)
(p/'audio-check.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2));assert a==b
