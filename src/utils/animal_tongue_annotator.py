# coding: utf-8
"""V0.6.5 local-only tongue polygon editor; no online content or detectors."""
import base64
from io import BytesIO
import json
from PIL import Image


def tongue_annotation_html(frames, source_hashes, roi):
    """Build self-contained HTML, requiring an explicit decision for EVERY frame."""
    data=[]
    for frame in frames:
        buf=BytesIO()
        Image.fromarray(frame['rgb'],'RGB').save(buf,format='PNG')
        data.append({'alpha':frame['alpha'], 'url':'data:image/png;base64,'+
                     base64.b64encode(buf.getvalue()).decode('ascii')})
    payload=json.dumps(data,ensure_ascii=False)
    hashes=json.dumps(source_hashes,sort_keys=True)
    roi_js=json.dumps(list(roi))
    # Use plain replacement (not .format) to avoid accidentally interpreting JS braces.
    page=r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>V0.6.5 Tongue Polygon Annotation</title>
<style>body{font:15px system-ui,Segoe UI,sans-serif;background:#101926;color:#f0f5ff;max-width:920px;margin:20px auto}
button{padding:9px 15px;margin:5px;border-radius:7px;border:0;cursor:pointer}button:focus{outline:2px solid #ffe16b}
canvas{display:block;width:512px;max-width:100%;border:2px solid #77869b;touch-action:none;background:#202020}
#status{margin:12px 0;color:#d5eeff}pre{white-space:pre-wrap} .warn{color:#ffd284}</style></head>
<body><h2>V0.6.5 舌头独立遮罩标注</h2>
<p>在<strong>每一张已经配准的参考图</strong>上，沿可见舌头边缘顺时针依次点击至少 3 个点（建议 8–20 点）。
不要把牙齿、下唇或整块口腔包含进去。没有可见舌头的帧，必须点击“本帧无舌头”。</p>
<p class="warn">这个工具只保存人工选中的图像坐标，并不自动识别舌头；标注不正确会引起分层伪影。</p>
<div id="status"></div><canvas id="image" width="512" height="512"></canvas>
<div><button id="prev">上一帧</button><button id="next">下一帧</button><button id="undo">撤回一个点</button>
<button id="clear">清空并重标</button><button id="absent">本帧无舌头</button><button id="confirm">确认本帧多边形</button></div>
<div><button id="download">下载 mouth_tongue_polygons_verified.json</button></div>
<pre id="info"></pre>
<script>
'use strict';
const frames=__FRAMES__;
const hashes=__HASHES__;
const roi=__ROI__;
const coords=frames.map(()=>[]), confirmed=frames.map(()=>false);
let index=0;
const canvas=document.getElementById('image'), ctx=canvas.getContext('2d');
const images=frames.map(f=>{const im=new Image();im.src=f.url;im.onload=draw;return im;});
function draw(){
  const im=images[index];ctx.clearRect(0,0,512,512);if(im.complete&&im.naturalWidth)ctx.drawImage(im,0,0,512,512);
  ctx.strokeStyle='#ffdb4d';ctx.lineWidth=2.5;ctx.strokeRect(roi[0]*512,roi[1]*512,(roi[2]-roi[0])*512,(roi[3]-roi[1])*512);
  const p=coords[index];
  if(p.length){ctx.beginPath();ctx.moveTo(p[0][0]*512,p[0][1]*512);
    for(let k=1;k<p.length;k++)ctx.lineTo(p[k][0]*512,p[k][1]*512);
    if(p.length>=3)ctx.closePath();ctx.strokeStyle='#27fff2';ctx.lineWidth=2.5;ctx.stroke();
    p.forEach((xy,k)=>{ctx.beginPath();ctx.arc(xy[0]*512,xy[1]*512,4,0,Math.PI*2);ctx.fillStyle='#ff2b5e';ctx.fill();
      ctx.fillStyle='#fff';ctx.font='12px sans-serif';ctx.fillText(String(k+1),xy[0]*512+7,xy[1]*512-7);});}
  document.getElementById('status').textContent=`参考帧 ${index+1}/${frames.length}  alpha=${frames[index].alpha} | 点数 ${p.length} | 状态 ${confirmed[index]?'已确认':'未确认'}`;
  document.getElementById('info').textContent=`已确认 ${confirmed.filter(Boolean).length}/${frames.length} 帧。黄色矩形为口部 ROI。点击仅在 ROI 内生效。`;
}
canvas.addEventListener('click',ev=>{const r=canvas.getBoundingClientRect();const x=(ev.clientX-r.left)/r.width,y=(ev.clientY-r.top)/r.height;
 if(!(x>roi[0]&&x<roi[2]&&y>roi[1]&&y<roi[3])){alert('必须在黄色 ROI 内标注');return;}
 if(coords[index].length>=60){alert('最多 60 个点');return;}
 coords[index].push([Number(x.toFixed(6)),Number(y.toFixed(6))]);confirmed[index]=false;draw();});
document.getElementById('prev').onclick=()=>{index=Math.max(0,index-1);draw();};
document.getElementById('next').onclick=()=>{index=Math.min(frames.length-1,index+1);draw();};
document.getElementById('undo').onclick=()=>{coords[index].pop();confirmed[index]=false;draw();};
document.getElementById('clear').onclick=()=>{coords[index]=[];confirmed[index]=false;draw();};
document.getElementById('absent').onclick=()=>{coords[index]=[];confirmed[index]=true;draw();};
document.getElementById('confirm').onclick=()=>{if(coords[index].length<3){alert('至少 3 点，或点击“本帧无舌头”');return;}
 confirmed[index]=true;draw();};
document.getElementById('download').onclick=()=>{
 if(confirmed.some(x=>!x)){alert('必须确认每一帧，包括没有舌头的帧');return;}
 if(!coords.some(p=>p.length>=3)){alert('必须至少有一帧包含可见舌头多边形');return;}
 const polygons={};frames.forEach((f,i)=>{polygons[String(f.alpha)]=coords[i];});
 const doc={schema_version:1,coordinate_system:'normalized_xy_on_512px_aligned_reference',
   verified:true,source_sha256:hashes,polygons:polygons,
   note:'Human clicked tongue polygons on aligned RGB512 references; [] means confirmed absence.'};
 const blob=new Blob([JSON.stringify(doc,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob);
 const a=document.createElement('a');a.href=url;a.download='mouth_tongue_polygons_verified.json';a.click();URL.revokeObjectURL(url);
};
draw();
</script></body></html>'''
    return page.replace('__FRAMES__',payload).replace('__HASHES__',hashes).replace('__ROI__',roi_js)
