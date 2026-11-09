
const R=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id);
let rows=[],index=0,timer=null;
function stop(){if(timer!==null)clearInterval(timer);timer=null;$('play').textContent='Phát'}
function show(){
  const row=rows[index];if(!row)return;
  $('text').textContent=row.text;
  $('source').textContent=row.language+' · '+row.source_title+' · dấu cuối: '+row.punctuation;
  $('count').textContent=(index+1)+'/'+rows.length;
  $('decision').textContent=row.prediction==='EOT'?'EOT — có thể nhường lượt':'NEOT — tiếp tục chờ';
  $('score').textContent='P(EOT) = '+row.prob_EOT.toFixed(5)+' · ngưỡng dev cố định = '+row.threshold.toFixed(5)+' · nhãn tham chiếu: chưa có';
}
function selectArticle(){stop();rows=R.rows.filter(row=>row.language===$('lang').value&&row.source_id===$('article').value);index=0;show()}
function selectLanguage(){
  stop();$('article').replaceChildren();
  const articles=new Map(R.rows.filter(row=>row.language===$('lang').value).map(row=>[row.source_id,row.source_title]));
  articles.forEach((title,id)=>{const option=document.createElement('option');option.value=id;option.textContent=title;$('article').append(option)});
  $('article').value=articles.keys().next().value||'';selectArticle();
}
$('lang').onchange=selectLanguage;$('article').onchange=selectArticle;
$('prev').onclick=()=>{stop();index=Math.max(0,index-1);show()};
$('next').onclick=()=>{stop();index=Math.min(rows.length-1,index+1);show()};
$('play').onclick=()=>{
  if(timer!==null){stop();return}
  if(index===rows.length-1)index=0;show();$('play').textContent='Dừng';
  timer=setInterval(()=>{if(index>=rows.length-1){stop();return}index++;show()},1200);
};
$('model').textContent=JSON.stringify(R.model,null,2);selectLanguage();
