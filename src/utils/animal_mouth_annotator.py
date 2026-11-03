# coding: utf-8
"""Offline, click-to-annotate paired V0.6.2 reference images, no network."""

import base64
import json
from io import BytesIO


def _png_data(image):
    from PIL import Image
    buff = BytesIO()
    Image.fromarray(image, 'RGB').save(buff, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buff.getvalue()).decode('ascii')


def annotation_html(closed, opened, template):
    """Generate a self-contained local HTML editor. All landmarks require manual clicks.

    The downloaded JSON is the only source of user-confirmed correspondences.
    """
    data = json.dumps(template, separators=(',', ':'), ensure_ascii=False).replace('<', '\\u003c')
    imgc = _png_data(closed)
    imgo = _png_data(opened)
    return r'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>LivePortrait V0.6.3 Mouth Landmark Annotator</title>
<style>body{font-family:system-ui,'Microsoft Yahei',sans-serif;max-width:1170px;margin:18px auto;padding:0 14px;background:#f5f5f5;color:#222}
main{display:flex;gap:14px;flex-wrap:wrap}section{background:white;border-radius:10px;padding:10px;max-width:548px}
canvas{width:min(100%,512px);height:auto;cursor:crosshair;border:1px solid #777;touch-action:none}
button{padding:9px 12px;margin:4px;border-radius:6px;cursor:pointer}button.active{background:#1864ab;color:white}
small{color:#555}#status{font-weight:bold;padding:8px}#download{background:#17834c;color:white}#download:disabled{background:#bbb}
</style></head><body><h2>LivePortrait V0.6.3 · 双状态嘴部对应点标注</h2>
<p>两个画布来自同一只狗，右边已做 V0.6.2 上半脸平移配准。依次选择每个标签，并在<strong>左、右两张图中分别点击同一个解剖位置</strong>。
不是自动识别；请仔细选择嘴角、上唇、下唇与下巴的可对应纹理。点击完整后才能导出 verified JSON。</p>
<div id="buttons"></div><div id="status"></div>
<main><section><h3>CLOSED · 闭嘴/小开口</h3><canvas id="c" width="512" height="512"></canvas></section>
<section><h3>OPEN ALIGNED · 张嘴</h3><canvas id="o" width="512" height="512"></canvas></section></main>
<p><button id="download" disabled>下载已确认的 mouth_landmarks_verified.json</button>
<small>下载后将 JSON 放到 LivePortrait 仓库中，再运行 --landmarks 路径。切换图片或 ROI 后需重新标注。</small></p>
<script>
const template=__TEMPLATE__;
const images={c:new Image(),o:new Image()};
images.c.src='__CLOSED__';images.o.src='__OPEN__';
const names=Object.keys(template.closed).sort();
const selections={c:{},o:{}};let active=0;
const canvases={c:document.getElementById('c'),o:document.getElementById('o')};
function redraw(){
  for(const side of ['c','o']){
    let cv=canvases[side], ctx=cv.getContext('2d');ctx.clearRect(0,0,512,512);
    if(images[side].complete&&images[side].naturalWidth)ctx.drawImage(images[side],0,0,512,512);
    const [x0,y0,x1,y1]=template.roi;ctx.lineWidth=2;ctx.strokeStyle='#fa3535';ctx.strokeRect(x0*512,y0*512,(x1-x0)*512,(y1-y0)*512);
    for(let i=0;i<names.length;i++){
       const name=names[i];let p=selections[side][name];if(!p)continue;
       const x=p[0]*512,y=p[1]*512;
       ctx.beginPath();ctx.arc(x,y,6,0,2*Math.PI);ctx.fillStyle=i===active?'#00ffcc':'#ffff00';ctx.fill();
       ctx.strokeStyle='#111';ctx.stroke();ctx.font='bold 15px sans-serif';ctx.lineWidth=3;ctx.strokeText(name,x+9,y-9);
       ctx.fillStyle='white';ctx.fillText(name,x+9,y-9);
    }
  }
  document.getElementById('status').textContent=`当前标注：${names[active]}；闭嘴已标 ${Object.keys(selections.c).length}/${names.length}，张嘴已标 ${Object.keys(selections.o).length}/${names.length}`;
  document.getElementById('download').disabled=names.some(n=>!selections.c[n]||!selections.o[n]);
  document.querySelectorAll('.point').forEach((b,i)=>b.classList.toggle('active',i===active));
}
const controls=document.getElementById('buttons');
for(let i=0;i<names.length;i++){
  let b=document.createElement('button');b.className='point';b.textContent=`${i+1}. ${names[i]}`;b.onclick=()=>{active=i;redraw()};controls.appendChild(b);
}
for(const side of ['c','o']){
  let cv=canvases[side];cv.addEventListener('click',event=>{
    const rect=cv.getBoundingClientRect();let x=(event.clientX-rect.left)/rect.width,y=(event.clientY-rect.top)/rect.height;
    const [x0,y0,x1,y1]=template.roi;
    if(!(x>x0+0.008&&x<x1-0.008&&y>y0+0.008&&y<y1-0.008)){
      alert('标注点必须位于红色嘴部 ROI 内侧。若范围不适合，请修改 --roi 并重新生成模板。');return;
    }
    selections[side][names[active]]=[Number(x.toFixed(6)),Number(y.toFixed(6))];
    if(selections.c[names[active]]&&selections.o[names[active]]&&active<names.length-1)active++;
    redraw();
  });
}
images.c.onload=redraw;images.o.onload=redraw;
document.getElementById('download').onclick=()=>{
 if(names.some(n=>!selections.c[n]||!selections.o[n]))return;
 const payload={...template,verified:true,closed:selections.c,open:selections.o};
 const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'});
 const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='mouth_landmarks_verified.json';a.click();
 setTimeout(()=>URL.revokeObjectURL(url),2000);
};redraw();
</script></body></html>'''.replace('__TEMPLATE__', data).replace('__CLOSED__', imgc).replace('__OPEN__', imgo)
