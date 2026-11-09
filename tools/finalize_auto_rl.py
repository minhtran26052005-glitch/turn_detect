"""Validate/review the bounded crawl-and-reward experiment, without promotion."""
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import LABEL2ID,file_hash,input_id,read_jsonl,write_json
from turn_v2.metrics import classification_metrics

ROOT=Path('outputs/auto_rl_v12_20261025')


def main():
    checks=[]
    def check(name,value):checks.append({'check':name,'passed':bool(value)})
    report=json.loads((ROOT/'report.json').read_text(encoding='utf-8'));identity=json.loads((ROOT/'identity.json').read_text(encoding='utf-8'))
    probe=list(read_jsonl(ROOT/'judge_probe.jsonl'));check('judge_probe6_passed',len(probe)==6 and all(r['judge_label']==r['expected'] for r in probe) and report['probe_passed'])
    check('loop_code_unchanged',identity['source_sha256']==file_hash(Path('turn_v2/auto_rl_v11.py')))
    launch=json.loads((ROOT/'launch_receipt.json').read_text(encoding='utf-8'));out=json.loads(''.join(o['text'] for o in launch['outputs'] if o['output_type']=='stream'))
    check('generated_judge_code_unchanged',out['source_sha256']==file_hash(Path('turn_v2/auto_rl_v12.py')))
    check('no_real_quality_overclaim',report['quality_target_certified'] is False)
    results=[];counts=[]
    for round_record in report.get('rounds',[]):
        name='round_%02d'%round_record['round'];folder=ROOT/name;audit=list(read_jsonl(folder/'judge_audit.jsonl'));candidates=list(read_jsonl(folder/'candidates.jsonl'))
        check(name+':candidate_count',len(candidates)==round_record['candidates']==len(audit))
        check(name+':candidate_labels_unassigned',all(r['label'] is None and r['is_gold'] is False for r in candidates))
        check(name+':judge_sees_same_visible_text',all(a['candidate_id']==c['candidate_id'] and a['text']==c['text'] and a['context']==c['context'] for a,c in zip(audit,candidates,strict=True)))
        check(name+':judge_acceptance_recomputed',all(r['judge_accepted']==(r['judge_labels'][0]==r['judge_labels'][1] and r['judge_labels'][0] in ['EOT','NEOT']) for r in audit))
        check(name+':source_hashes',all(hashlib.sha256(r['text'].encode()).hexdigest()==r['sha256'] for r in read_jsonl(folder/'sources.jsonl')))
        for lang in ['vi','en','zh']:
            p=folder/lang;item=json.loads((p/'report.json').read_text(encoding='utf-8'));accepted=[r for r in audit if r['language']==lang and r['judge_accepted']]
            actual={label:sum(r['judge_label']==label for r in accepted) for label in ['EOT','NEOT']};stored=item['judge_counts']
            check(name+':'+lang+':accepted_counts',all(actual[k]==stored.get(k,0) for k in actual))
            counts.append({'round':round_record['round'],'language':lang,'accepted_EOT':actual['EOT'],'accepted_NEOT':actual['NEOT']})
            dev=list(read_jsonl(Path('data_v10/reward_20261025')/lang/'dev.jsonl'))
            check(name+':'+lang+':no_exact_dev_overlap',not {input_id(r) for r in accepted}&{input_id(r) for r in dev})
            if item['status']=='AUTO_CRAWL_JUDGE_RL_ROUND_COMPLETE':
                prediction=np.load(p/'dev_predictions.npz');labels=np.array([LABEL2ID[r['label']] for r in dev]);check(name+':'+lang+':dev_labels',np.array_equal(labels,prediction['labels']))
                for source,expected in item['dev_after']['slices'].items():
                    mask=np.array([r['evaluation_slice']==source for r in dev]);check(name+':'+lang+':'+source,classification_metrics(labels[mask],prediction['prob_EOT'][mask],item['dev_after']['threshold'])==expected)
                check(name+':'+lang+':dev_gate',item['candidate_retained']==(item['dev_after']['selection_score']>item['dev_before']['selection_score']))
                results.append({'round':round_record['round'],'language':lang,'steps':item['steps'],'kept':item['candidate_retained'],
                    'accuracy_before':item['dev_before']['slices']['natural']['accuracy'],'F1_before':item['dev_before']['slices']['natural']['eot_f1'],
                    'accuracy_after':item['dev_after']['slices']['natural']['accuracy'],'F1_after':item['dev_after']['slices']['natural']['eot_f1'],
                    'min_score_before':item['dev_before']['selection_score'],'min_score_after':item['dev_after']['selection_score']})
            else:check(name+':'+lang+':skip_unbalanced_rewards',min(actual.values())<2 and item['updated'] is False)
    state=json.loads((ROOT/'jobs/auto_rl_v12_20261025/status.json').read_text(encoding='utf-8'));check('Colab_job_completed',state['status']=='COMPLETED' and state['exit_code']==0)
    old=json.loads(Path('outputs/auto_rl_v11_20261025/report.json').read_text(encoding='utf-8'));check('failed_judge_did_not_update_policy',old['status']=='JUDGE_PROBE_FAILED_NO_POLICY_UPDATES' and old['updates']==0)
    summary={'status':report['status'],'quality_target_certified':False,'rounds':report.get('rounds',[]),'reward_label_counts':counts,'results':results,'selected_models':report.get('selected_models'),
        'GPU_peak_GiB':report.get('GPU_peak_bytes',0)/2**30,'local_disk_free_GiB':shutil.disk_usage(ROOT).free/2**30,'fresh_test_evaluated':False,
        'scope':'Automatic semantic judge reward, one-step RL, natural-floor dev gate; judge reward and real-turn reference are different targets. Not a90%certificate.'}
    write_json(ROOT/'summary.json',summary)
    table=''.join('<tr><td>'+str(v['round'])+'</td><td>'+v['language']+'</td><td>'+str(v['steps'])+'</td><td>'+f"{v['accuracy_before']:.2%} / {v['F1_before']:.2%}"+'</td><td>'+f"{v['accuracy_after']:.2%} / {v['F1_after']:.2%}"+'</td><td>'+('giữ' if v['kept'] else 'bỏ nhánh')+'</td></tr>' for v in results)
    page='<!doctype html><html lang="vi"><meta charset="utf-8"><title>Auto crawl RL</title><style>body{font:18px system-ui;max-width:1150px;margin:35px auto;padding:24px;color:#18313e;background:#f3f7fb}p{line-height:1.7}td,th{padding:12px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{background:white;padding:24px;border-radius:12px;margin:18px 0}</style><h1>Crawl → dự đoán → reward Qwen → RL → dev</h1><p>Đã chạy tự động2vòng, ba model riêng. Qwen3-8B dùng làm người chấm semantic EOT/NEOT/AMBIGUOUS; model nhỏ mới là policy. Reward do Qwen chấm không phải nhãn người duyệt hoặc thời gian nhường lượt thật. Không chứng nhận90%.</p><section><h2>Các nhánh đã chạy</h2><table><tr><th>Vòng</th><th>Tiếng</th><th>Bước RL</th><th>Dev trước: acc/F1</th><th>Dev sau: acc/F1</th><th>Dev gate</th></tr>'+table+'</table><p>Chỉ giữ khi minimum dev-slice score tăng; không dùng tập regression384để tune. Reward judge và natural-floor dev là hai mục tiêu khác nhau.</p></section><section><h2>Crawl và reward được nhận</h2><pre>'+html.escape(json.dumps({'rounds':summary['rounds'],'reward_labels':counts},ensure_ascii=False,indent=2))+'</pre><p>Qwen trả hai đáp án độc lập theo prompt nhưng cùng một model, nên vẫn có tương quan. Mẫu bất đồng/AMBIGUOUS/lỗi định dạng không tạo reward.</p></section><section><h2>Bộ kiểm tra người chấm</h2><p>Lượt4B theo relativeABC không vượt kiểm tra, không cập nhật policy. Lượt8B dùng đáp án sinh trực tiếp, vượt6/6probe rõ ràng. Thay cả model và cách đọc đáp án; không kết luận riêng tác động của tăng số tham số. Probe nhỏ không phải chứng nhận độ chính xác trên crawl.</p><pre>'+html.escape(json.dumps([{'language':r['language'],'text':r['text'],'expected':r['expected'],'generated':r['judge_labels']} for r in probe],ensure_ascii=False,indent=2))+'</pre><p><a href="summary.json">JSON kết quả</a> · <a href="../reward_v10_20261025/review.html">Đối chứng CE/cost-CE/RL V10</a> · <a href="../../README_AUTO_RL.md">Cách chạy và giới hạn</a></p></section></html>'
    (ROOT/'review.html').write_text(page,encoding='utf-8')
    write_json(ROOT/'completion_validation.json',{'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','passed':sum(c['passed'] for c in checks),'total':len(checks),'checks':checks,'quality_target_certified':False})
    print(json.dumps(summary),flush=True)
    if not all(c['passed'] for c in checks):print(json.dumps([c for c in checks if not c['passed']]));raise SystemExit(1)


if __name__=='__main__':main()
