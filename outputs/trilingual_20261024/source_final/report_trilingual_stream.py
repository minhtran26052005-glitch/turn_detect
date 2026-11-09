"""Standalone interactive review of every observed streaming prefix."""
import json
from pathlib import Path


def main():
    root = Path("outputs/trilingual_stream_20261024")
    models = {}
    for name, relative in [("Việt QAT cũ", "vi_qat_baseline"), ("Student Việt–Anh–Trung INT4", "trilingual_student")]:
        folder = root / relative
        if not (folder / "trace.jsonl").exists():continue
        trace = [json.loads(line) for line in (folder / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line]
        models[name] = {"trace": trace, "paragraphs": json.loads((folder / "paragraphs.json").read_text(encoding="utf-8")),
                        "report": json.loads((folder / "report.json").read_text(encoding="utf-8"))}
    assert models
    encoded = json.dumps(models, ensure_ascii=False).replace("<", "\\u003c")
    html = r'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Streaming turn detection — Việt / English / 中文</title>
<style>body{font:16px system-ui;background:#f4f7fa;color:#152534;margin:0}main{max-width:1120px;margin:30px auto;padding:22px}h1{font-size:27px;margin:0 0 10px}.box{background:white;border:1px solid #d7e0e8;border-radius:12px;padding:20px;margin:16px 0}.note{color:#526372;line-height:1.6}label{display:inline-block;margin:8px 15px 8px 0}select,button{font:inherit;border:1px solid #b8c8d5;border-radius:6px;padding:8px;background:white}button{cursor:pointer}input[type=range]{width:100%}#prefix{white-space:pre-wrap;font-size:22px;line-height:1.7;min-height:120px}#result{font-size:22px;font-weight:700}canvas{width:100%;height:210px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:9px;border-bottom:1px solid #e2e9ef}a{color:#145da0}.danger{color:#a63325}.good{color:#087461}#source{font-size:14px}small{color:#526372}.scroll{max-height:350px;overflow:auto}</style>
<main><h1>Phát lại văn bản và xem từng dự đoán EOT</h1>
<p class="note">Văn bản crawl có nguồn. Phát từng từ ở Việt/Anh và từng ký tự Hán ở Trung. <b>Dấu phẩy/dấu chấm không phải nhãn chuẩn kết thúc lượt nói.</b> Đây là chẩn đoán văn xuôi, không phải accuracy/F1 hội thoại hay kiểm tra VAD/audio.</p>
<section class="box"><label>Model <select id="model"></select></label><label>Ngôn ngữ <select id="language"><option value="vi">Tiếng Việt</option><option value="en">English</option><option value="zh">中文</option></select></label><label>Đoạn <select id="paragraph"></select></label>
<div id="source"></div><p><button id="play">▶ Phát</button> <button id="prev">←</button> <button id="next">→</button> <label>Tốc độ <select id="speed"><option value="400">Chậm</option><option value="150" selected>Vừa</option><option value="40">Nhanh</option></select></label></p>
<input id="slider" type="range" min="0" max="1" value="0"><p id="result"></p><p id="detail" class="note"></p><div id="prefix"></div></section>
<section class="box"><canvas id="chart" width="1040" height="210"></canvas><small>Đường xanh: xác suất EOT; đường đỏ: ngưỡng của ngôn ngữ; chấm cam: dấu phẩy/ngắt mệnh đề; đường đứng: vị trí đang xem.</small></section>
<section class="box"><h2 style="font-size:20px">Các điểm ngắt trong đoạn</h2><div class="scroll"><table><thead><tr><th>Bước</th><th>Dấu</th><th>P(EOT)</th><th>Dự đoán</th><th>Vẫn còn văn bản?</th><th>Input bị cắt?</th></tr></thead><tbody id="boundaries"></tbody></table></div></section>
<p class="note">Model bỏ dấu câu trước suy luận. Nếu chỉ thêm dấu phẩy/chấm ở cuối prefix, xác suất thường không đổi. Cột input bị cắt đánh dấu phần văn bản vượt giới hạn token; bản hiện tại giữ phần đầu, nên có thể bỏ lỡ từ mới thêm. <a href="../trilingual_20261024/review.html">Xem dữ liệu, teacher và kết quả pilot</a>.</p></main>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=x=>document.getElementById(x);let points=[],timer=null;
for(const name of Object.keys(D))$('model').add(new Option(name,name));
function stop(){clearInterval(timer);timer=null;$('play').textContent='▶ Phát'}
function choose(){stop();const m=D[$('model').value];$('paragraph').replaceChildren();for(const p of m.paragraphs.filter(x=>x.language===$('language').value))$('paragraph').add(new Option(p.title+' — đoạn '+p.paragraph_id,p.paragraph_id));load()}
function load(){stop();const m=D[$('model').value],id=Number($('paragraph').value);points=m.trace.filter(x=>x.paragraph_id===id);$('slider').max=Math.max(0,points.length-1);$('slider').value=0;const p=m.paragraphs.find(x=>x.paragraph_id===id);$('source').replaceChildren();if(p){const a=document.createElement('a');a.href=p.source_url;a.target='_blank';a.rel='noopener';a.textContent=p.title+' — nguồn Wikipedia';$('source').append(a)}$('boundaries').replaceChildren();points.forEach((p,i)=>{if(p.boundary_kind==='content')return;const tr=document.createElement('tr');for(const v of [p.step,p.emitted_piece,p.prob_EOT.toFixed(4),p.prediction,p.has_observed_continuation?'Có':'Không',p.input_truncated?'Có':'Không']){const td=document.createElement('td');td.textContent=v;tr.append(td)}tr.style.cursor='pointer';tr.onclick=()=>{$('slider').value=i;draw()};$('boundaries').append(tr)});draw()}
function draw(){const i=Number($('slider').value),p=points[i];if(!p)return;$('prefix').textContent=p.raw_prefix;$('result').textContent=p.prediction+' · P(EOT) '+p.prob_EOT.toFixed(4)+' · ngưỡng '+p.threshold.toFixed(3);$('result').className=p.prediction==='EOT'?'good':'';$('detail').textContent='Bước '+(i+1)+'/'+points.length+' · token trước cắt: '+p.input_tokens_before_truncation+' · điểm: '+p.boundary_kind+(p.input_truncated?' · INPUT BỊ CẮT: phần mới có thể không còn được quan sát':'');$('detail').className=p.input_truncated?'danger':'note';const c=$('chart'),x=c.getContext('2d'),w=c.width,h=c.height,L=42,R=w-15,T=15,B=h-28; x.clearRect(0,0,w,h);x.strokeStyle='#dce5ed';x.fillStyle='#526372';x.font='12px system-ui';for(const q of [0,.25,.5,.75,1]){const y=B-q*(B-T);x.beginPath();x.moveTo(L,y);x.lineTo(R,y);x.stroke();x.fillText(q.toFixed(2),3,y+4)}const xx=j=>L+j/Math.max(1,points.length-1)*(R-L),yy=q=>B-q*(B-T);x.strokeStyle='#c9473c';x.setLineDash([6,5]);x.beginPath();x.moveTo(L,yy(p.threshold));x.lineTo(R,yy(p.threshold));x.stroke();x.setLineDash([]);x.strokeStyle='#167b83';x.lineWidth=2;x.beginPath();points.forEach((q,j)=>j?x.lineTo(xx(j),yy(q.prob_EOT)):x.moveTo(xx(j),yy(q.prob_EOT)));x.stroke();for(let j=0;j<points.length;j++){if(points[j].boundary_kind==='comma_or_clause'){x.fillStyle='#dc8418';x.beginPath();x.arc(xx(j),yy(points[j].prob_EOT),3,0,7);x.fill()}}x.strokeStyle='#384a5b';x.beginPath();x.moveTo(xx(i),T);x.lineTo(xx(i),B);x.stroke()}
$('model').onchange=choose;$('language').onchange=choose;$('paragraph').onchange=load;$('slider').oninput=draw;$('prev').onclick=()=>{$('slider').value=Math.max(0,Number($('slider').value)-1);draw()};$('next').onclick=()=>{$('slider').value=Math.min(points.length-1,Number($('slider').value)+1);draw()};$('play').onclick=()=>{if(timer){stop();return}$('play').textContent='⏸ Dừng';timer=setInterval(()=>{const i=Number($('slider').value);if(i>=points.length-1){stop();return}$('slider').value=i+1;draw()},Number($('speed').value))};$('speed').onchange=stop;choose();
</script></html>'''.replace("__DATA__", encoded)
    (root / "review.html").write_text(html, encoding="utf-8")
    print(json.dumps({"review": str(root / "review.html"), "models": list(models), "bytes": len(html.encode("utf-8"))}))


if __name__ == "__main__": main()
