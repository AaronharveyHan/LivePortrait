# coding: utf-8
"""Offline local HTML annotator for intermediate dog mouth correspondences."""
import base64
import io
import json

import numpy as np

from .animal_mouth_geometry import check_image


def _data_uri(rgb):
    from PIL import Image
    stream=io.BytesIO()
    Image.fromarray(check_image(rgb,'annotation source'),'RGB').save(stream,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')


def annotation_html(frames, names, roi, image_hashes):
    """frames = [(alpha, aligned_512_RGB, proxy_points_px), ...].

    A click is required for EVERY point in EVERY frame before export. Proxy dots
    are shown as guides only. No remote assets, data transfer or autosave.
    """
    if not frames:
        raise ValueError('At least one middle frame required for annotation')
    data={
        'names':list(names),'roi':list(roi),
        'frames':[{'alpha':format(float(a),'.9g'),'image':_data_uri(img),
                   'proxy':[[round(float(v)/512,6) for v in xy] for xy in points]}
                  for a,img,points in frames],
        'hashes':dict(image_hashes),
    }
    payload=json.dumps(data,ensure_ascii=True).replace('</','<\\/')
    return r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>V0.6.4 – middle mouth landmarks</title>
<style>body{font-family:system-ui,sans-serif;background:#111820;color:#eff6ff;max-width:1000px;margin:25px auto;padding:0 18px}h1{font-size:24px}p{line-height:1.6;color:#b9cada}canvas{width:min(100%,620px);height:auto;cursor:crosshair;border:1px solid #789;background:#222}button,select{font-size:15px;padding:8px;margin:4px;background:#294760;border:1px solid #6a90a7;border-radius:5px;color:white;cursor:pointer}.active{background:#19826f}#status{font-weight:bold;padding:10px 0}code{color:#a9e9c7}</style></head><body>
<h1>V0.6.4 · Intermediate mouth landmarks</h1>
<p>Same dog, aligned 512×512 frames. Click each landmark on each REAL intermediate reference frame. <b>Proxy dots are suggestions, NOT human-verified.</b> Left/right mean IMAGE coordinates. Never mark tongue tip as lower lip/chin. File is offline; images do not leave your browser.</p>
<div>Frame: <select id="frame"></select></div><div id="names"></div>
<canvas id="view" width="512" height="512"></canvas>
<div id="status"></div>
<button id="export">Export mouth_mid_landmarks_verified.json</button>
<button id="reset">Clear current frame clicks</button>
<p>All five points on EVERY frame must be clicked before export. Download the JSON and pass it via <code>--mid-points</code> to run_animal_mouth_multiref.py.</p>
<script type="application/json" id="payload">'''+payload+r'''</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('payload').textContent);
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
const chooser=document.getElementById('frame'),buttons=document.getElementById('names');
const status=document.getElementById('status');
let current=0,pointIndex=0;
const imgs=data.frames.map(f=>{const img=new Image();img.src=f.image;return img});
const picked=data.frames.map(f=>f.proxy.map(p=>({xy:p.slice(),clicked:false})));
for(let i=0;i<data.frames.length;i++){const o=document.createElement('option');o.value=i;o.textContent=`Frame ${i+1} (alpha=${data.frames[i].alpha})`;chooser.append(o)}
chooser.onchange=()=>{current=Number(chooser.value);pointIndex=0;draw()};
function draw(){
  const f=data.frames[current];
  ctx.fillStyle='#141e25';ctx.fillRect(0,0,512,512);
  if(imgs[current].complete&&imgs[current].naturalWidth)ctx.drawImage(imgs[current],0,0,512,512);
  const [x0,y0,x1,y1]=data.roi;ctx.strokeStyle='#ff6666';ctx.lineWidth=2;ctx.strokeRect(x0*512,y0*512,(x1-x0)*512,(y1-y0)*512);
  buttons.textContent='';
  data.names.forEach((name,i)=>{
    const p=picked[current][i],x=p.xy[0]*512,y=p.xy[1]*512;
    ctx.beginPath();ctx.arc(x,y,i===pointIndex?7:5,0,2*Math.PI);
    ctx.fillStyle=p.clicked?'#27d895':'#ffd469';ctx.fill();ctx.strokeStyle='#102535';ctx.lineWidth=2;ctx.stroke();
    ctx.font='13px sans-serif';ctx.lineWidth=3;ctx.strokeStyle='#000';ctx.strokeText(`${i+1}. ${name}`,x+10,y-8);
    ctx.fillStyle='#fff';ctx.fillText(`${i+1}. ${name}`,x+10,y-8);
    const b=document.createElement('button');b.textContent=`${i+1}. ${name}${p.clicked?' ✓':' ?'}`;
    if(i===pointIndex)b.className='active';b.onclick=()=>{pointIndex=i;draw()};buttons.append(b);
  });
  const done=picked.flat().filter(p=>p.clicked).length,total=data.frames.length*data.names.length;
  status.textContent=`Clicked ${done}/${total} points. Select ${data.names[pointIndex]} on frame ${current+1}.`;
}
canvas.onclick=(ev)=>{
  const rect=canvas.getBoundingClientRect();
  const x=(ev.clientX-rect.left)/rect.width,y=(ev.clientY-rect.top)/rect.height;
  const [x0,y0,x1,y1]=data.roi;
  if(!Number.isFinite(x)||!Number.isFinite(y)||!(x>x0+.008&&x<x1-.008&&y>y0+.008&&y<y1-.008)){
    alert('Click must be strictly inside the red ROI.');return;
  }
  picked[current][pointIndex]={xy:[Number(x.toFixed(6)),Number(y.toFixed(6))],clicked:true};
  const next=picked[current].findIndex(p=>!p.clicked);
  if(next>=0)pointIndex=next;
  else if(current<data.frames.length-1){current++;chooser.value=current;pointIndex=0}
  draw();
};
document.getElementById('reset').onclick=()=>{picked[current].forEach(p=>p.clicked=false);pointIndex=0;draw()};
document.getElementById('export').onclick=()=>{
  if(picked.some(fr=>fr.some(p=>!p.clicked))){alert('Please click ALL points on EVERY frame. Proxy dots are unverified.');return}
  const frames={};
  data.frames.forEach((f,i)=>{const points={};data.names.forEach((name,j)=>points[name]=picked[i][j].xy);frames[f.alpha]=points});
  const manifest={schema_version:1,coordinate_system:'normalized_xy_on_512px_aligned_reference',
    verified:true,frames:frames,source_sha256:data.hashes,
    note:'Each landmark was clicked by a human on the registered intermediate frame; animal identity is user asserted.'};
  const blob=new Blob([JSON.stringify(manifest,null,2)+'\n'],{type:'application/json'});
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='mouth_mid_landmarks_verified.json';a.click();
  setTimeout(()=>URL.revokeObjectURL(url),2000);
};
imgs.forEach(img=>img.onload=draw);draw();
</script></body></html>'''
