// Verify local report behavior without claiming browser layout inspection.
const fs=require('fs');
const vm=require('vm');
const assert=require('assert');
const root='outputs/natural_v7_20261025';
function inspect(path){
  const html=fs.readFileSync(path,'utf8');
  const payload=html.match(/<script[^>]*id="data"[^>]*>([\s\S]*?)<\/script>/)[1];
  const data=JSON.parse(payload);
  const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
  const script=scripts[scripts.length-1][1];
  const nodes={};
  function element(){return {textContent:'',value:'',children:[],append(x){this.children.push(x)},replaceChildren(){this.children=[]}}}
  const document={getElementById(id){if(!nodes[id])nodes[id]=element();return nodes[id]},createElement(){return element()}};
  document.getElementById('data').textContent=payload;
  document.getElementById('lang').value='all';
  let timerCallback=null;
  vm.runInNewContext(script,{document,JSON,Object,
    setInterval(fn){timerCallback=fn;return 1},clearInterval(){timerCallback=null}},{timeout:1000});
  return {data,nodes,tick(){if(timerCallback)timerCallback()}};
}
const report=inspect(root+'/review.html');
assert.equal(report.nodes.rows.children.length,report.data.rows.length);
report.nodes.lang.value='vi';report.nodes.lang.onchange();
assert.equal(report.nodes.rows.children.length,report.data.rows.filter(r=>r.language==='vi').length);
if(report.data.final_held)assert.equal(report.nodes.heldRows.children.length,3);
const audio=inspect(root+'/asr_review/review.html');
assert.equal(audio.data.length,20);
assert.equal(audio.nodes.audio.src,audio.data[0].audio);
audio.nodes.next.onclick();assert.equal(audio.nodes.audio.src,audio.data[1].audio);
assert.equal(audio.nodes.new.textContent,audio.data[1].text_new);
audio.nodes.prev.onclick();assert.equal(audio.nodes.audio.src,audio.data[0].audio);
const result={status:'PASS',scope:'Local JavaScript behavior with a minimal DOM stub; not browser visual QA',
  development_rows:report.data.rows.length,language_filter:true,held_table:!!report.data.final_held,audio_cases:audio.data.length,
  audio_next_previous:true};
if(report.data.final_held){
  const crawl=inspect(root+'/crawl_review.html');
  assert.equal(crawl.data.rows.length,204);
  assert(crawl.data.rows.every(row=>row.label===null&&row.is_gold===false));
  for(const lang of ['vi','en','zh']){
    crawl.nodes.lang.value=lang;crawl.nodes.lang.onchange();
    const first=crawl.data.rows.find(row=>row.language===lang);
    assert.equal(crawl.nodes.text.textContent,first.text);
    const article=crawl.data.rows.filter(row=>row.language===lang&&row.source_id===first.source_id);
    assert(article.length>1);
    crawl.nodes.next.onclick();assert.equal(crawl.nodes.text.textContent,article[1].text);
    crawl.nodes.prev.onclick();assert.equal(crawl.nodes.text.textContent,article[0].text);
    crawl.nodes.play.onclick();crawl.tick();assert.equal(crawl.nodes.text.textContent,article[1].text);
    crawl.nodes.play.onclick();assert.equal(crawl.nodes.play.textContent,'Phát');
  }
  result.crawl_cases=204;result.crawl_language_filter=true;result.crawl_saved_prefix_replay=true;
}
fs.writeFileSync(root+'/review_behaviour_validation.json',JSON.stringify(result,null,2));
process.stdout.write(JSON.stringify(result)+'\n');
