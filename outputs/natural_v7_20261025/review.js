
const R=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id);
function line(target,values){
  const tr=document.createElement('tr');
  values.forEach(text=>{const td=document.createElement('td');td.textContent=text;tr.append(td)});
  $(target).append(tr);
}
function percent(value){return (value*100).toFixed(2)+'%'}
function show(){
  $('rows').replaceChildren();
  R.rows.filter(r=>$('lang').value==='all'||r.language===$('lang').value).forEach(r=>
    line('rows',[r.model,r.language,percent(r.accuracy),percent(r.eot_f1),percent(r.false_eot_rate),r.scope]));
}
$('lang').onchange=show;
$('asr').textContent=JSON.stringify(R.ASR_diagnostics||{status:'Chưa thu đủ kết quả PhoWhisper'},null,2);
$('status').textContent=JSON.stringify({status:R.status,held_cases:R.held_cases,
  held_classifier_scored:R.held_classifier_scored,crawl_gold_labels:R.crawl_gold_labels,
  crawl_accuracy:R.crawl_accuracy,user_target_achieved:R.user_target_achieved},null,2);
if(R.final_held){
  $('heldMeta').textContent=JSON.stringify({model:R.final_held.selected_kind,path:R.final_held.selected_path,
    calls:R.final_held.held_calls,note:'Nhãn tự động; cohort nhỏ, chưa có chứng nhận human gold'},null,2);
  Object.entries(R.final_held.metrics.selected).forEach(([lang,r])=>
    line('heldRows',[lang,r.samples,percent(r.accuracy),percent(r.eot_f1),percent(r.false_eot_rate)]));
  if(R.final_held.automatic_held_numeric_target_met)
    $('headline').textContent='Đạt ngưỡng số trên cohort nhãn tự động; chưa có human gold.';
}else{
  $('heldMeta').textContent='Chưa chấm classifier: giữ riêng để tránh tune vào test.';
}
show();
