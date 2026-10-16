// Dashboard page served by the ESP32-S3 itself (no internet needed).
#pragma once
const char DASH_HTML[] PROGMEM = R"rawliteral(<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>PBEL Noise Monitor</title><style>
:root{--bg:#f4f6fa;--card:#fff;--tx:#1b2433;--mu:#6b7686;--bd:#e0e5ee;--ac:#2563eb;--ok:#16a34a;--er:#dc2626}
@media(prefers-color-scheme:dark){:root{--bg:#0f1520;--card:#18212f;--tx:#e8edf5;--mu:#8e9bb0;--bd:#263246;--ac:#60a5fa}}
*{box-sizing:border-box}body{margin:0;font:15px system-ui,sans-serif;background:var(--bg);color:var(--tx)}
header{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;background:var(--card);border-bottom:1px solid var(--bd)}
h1{font-size:17px;margin:0}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--er);margin-right:6px}.dot.on{background:var(--ok)}
main{display:grid;gap:12px;padding:12px;max-width:1000px;margin:auto;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px}.wide{grid-column:1/-1}
h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mu);margin:0 0 10px}
.big{font-size:26px;font-weight:700}.sub{color:var(--mu);font-size:13px}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;font-weight:600;color:#fff;background:var(--mu)}
.row{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:13px}.row span:first-child{width:140px}
.bar{flex:1;height:10px;background:var(--bd);border-radius:5px;overflow:hidden}.bar i{display:block;height:100%;width:0;background:var(--ac);transition:width .25s}
canvas{width:100%;height:150px;display:block}table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--bd)}
.st{display:grid;grid-template-columns:1fr 1fr;gap:8px 12px;font-size:12px;color:var(--mu)}.st b{display:block;font-size:17px;color:var(--tx)}
button{background:var(--ac);color:#fff;border:0;border-radius:8px;padding:6px 12px;cursor:pointer;float:right}
</style></head><body>
<header><h1>PBEL Acoustic Event Monitor</h1><div class=sub><span id=dot class=dot></span><span id=conn>connecting</span> &middot; up <span id=up>-</span></div></header>
<main>
<section class=card><h2>Current status</h2><div id=state class=big>--</div><div id=sub class=sub>&nbsp;</div><p><span id=prio class=badge>-</span></p></section>
<section class=card><h2>Class probabilities (last window)</h2><div id=probs></div></section>
<section class="card wide"><h2>Sound level (relative RMS) &middot; dashed: trigger on / off</h2><canvas id=cv></canvas></section>
<section class=card><h2>Events since boot</h2><div id=cnt class=st></div></section>
<section class=card><h2>System</h2><div id=sys class=st></div></section>
<section class="card wide"><h2>Recent confirmed events<button onclick=csv()>Download CSV</button></h2>
<table><thead><tr><th>#</th><th>Time</th><th>Source</th><th>Confidence</th><th>Duration</th><th>Priority</th></tr></thead><tbody id=ev><tr><td colspan=6>No events yet</td></tr></tbody></table></section>
</main><p class=sub style="text-align:center">Levels are relative RMS, not calibrated dB SPL. Classification runs on the ESP32-S3 without internet.</p>
<script>
const CL=["Normal Ambient","Vehicle Horn","Construction / Drilling","Emergency Siren"],PR=["-","Disturbance","Disturbance","Alert"],PC={"-":"#6b7686",Disturbance:"#d97706",Alert:"#dc2626"};
const $=i=>document.getElementById(i);let H=[],E=[],seen=-1,fails=0;
$('probs').innerHTML=CL.map((c,i)=>'<div class=row><span>'+c+'</span><div class=bar><i id=p'+i+'></i></div><span id=pv'+i+' style="width:38px;text-align:right">0%</span></div>').join('');
function up(s){return Math.floor(s/3600)+'h '+Math.floor(s%3600/60)+'m '+s%60+'s'}
function draw(on,off){const c=$('cv'),r=window.devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*r;c.height=h*r;const x=c.getContext('2d');x.setTransform(r,0,0,r,0,0);
const cs=getComputedStyle(document.documentElement),mx=Math.max(on*2,...H,0.001)*1.1,Y=v=>h-4-(h-8)*v/mx;
x.strokeStyle=cs.getPropertyValue('--bd');x.strokeRect(0,0,w,h);
[[on,'#dc2626'],[off,'#d97706']].forEach(a=>{x.strokeStyle=a[1];x.setLineDash([5,4]);x.beginPath();x.moveTo(0,Y(a[0]));x.lineTo(w,Y(a[0]));x.stroke()});
x.setLineDash([]);x.strokeStyle=cs.getPropertyValue('--ac');x.lineWidth=2;x.beginPath();H.forEach((v,i)=>{const px=i*w/119;i?x.lineTo(px,Y(v)):x.moveTo(px,Y(v))});x.stroke()}
function evs(){fetch('/api/events').then(r=>r.json()).then(a=>{E=a;if(a.length)$('ev').innerHTML=a.map(e=>'<tr><td>'+e.id+'</td><td>'+e.time+'</td><td>'+e.source+'</td><td>'+Math.round(e.confidence*100)+'%</td><td>'+e.duration_s.toFixed(1)+' s</td><td><span class=badge style="background:'+(PC[e.priority]||PC['-'])+'">'+e.priority+'</span></td></tr>').join('')}).catch(()=>{})}
function csv(){let t='id,time,source,confidence,duration_s,priority\n'+E.map(e=>[e.id,e.time,e.source,e.confidence,e.duration_s,e.priority].join(',')).join('\n');
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([t],{type:'text/csv'}));a.download='pbel_events.csv';a.click()}
function render(d){H.push(d.rms);if(H.length>120)H.shift();
if(d.s&&d.cf>=0){$('state').textContent=CL[d.cf];$('sub').textContent='Confirmed event in progress';$('prio').textContent=PR[d.cf];$('prio').style.background=PC[PR[d.cf]]}
else if(d.s){$('state').textContent='Analysing...';$('sub').textContent='Event detected, confirming (3 of 5 windows)';$('prio').textContent='-';$('prio').style.background=PC['-']}
else{$('state').textContent='Listening';$('sub').textContent='Level below trigger';$('prio').textContent='-';$('prio').style.background=PC['-']}
d.p.forEach((v,i)=>{$('p'+i).style.width=Math.round(v*100)+'%';$('pv'+i).textContent=Math.round(v*100)+'%'});
$('cnt').innerHTML=CL.map((c,i)=>'<div>'+c+'<b>'+d.c[i]+'</b></div>').join('');
$('sys').innerHTML='<div>Inference<b>'+(d.ti/1000).toFixed(1)+' ms</b></div><div>Features<b>'+(d.tf/1000).toFixed(1)+' ms</b></div><div>Free heap<b>'+Math.round(d.heap/1024)+' KB</b></div><div>Confidence limit<b>'+Math.round(d.th*100)+'%</b></div><div>RMS now<b>'+d.rms.toFixed(4)+'</b></div><div>Windows<b>'+d.win+'</b></div>';
$('up').textContent=up(d.up);draw(d.on,d.off);if(d.n!==seen){seen=d.n;evs()}}
async function poll(){try{const r=await fetch('/api/live',{cache:'no-store'});render(await r.json());fails=0;$('dot').className='dot on';$('conn').textContent='live'}
catch(e){if(++fails>2){$('dot').className='dot';$('conn').textContent='disconnected'}}setTimeout(poll,300)}
poll();
</script></body></html>)rawliteral";
