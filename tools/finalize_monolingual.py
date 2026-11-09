"""Inspect three private models, freeze them, then compare seen regressions."""
import html
import json
from pathlib import Path
import sys
import shutil
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import LABEL2ID,file_hash,read_jsonl,write_json,write_jsonl
from turn_v2.metrics import classification_metrics
from turn_v2.monolingual_runtime import SeparateLanguageModels

ROOT=Path('outputs/monolingual_20261025')


def main():
    registry=json.loads((ROOT/'registry.json').read_text());checks=[]
    def check(name,condition):
        checks.append({'check':name,'passed':bool(condition)})
        if not condition:raise AssertionError(name)
    for language in ['en','zh']:
        teacher=ROOT/f'runs/mono_{language}_teacher_20261025';student=ROOT/f'runs/mono_{language}_student_20261025'
        report=json.loads((student/'monolingual_report.json').read_text());identity=json.loads((teacher/'identity.json').read_text())
        check(language+':teacher_only_one_language',identity['train_languages']==identity['dev_languages']==[language] and identity['shared_trainable_weights'] is False)
        check(language+':teacher_backup',file_hash(teacher/'best/model.safetensors')==report['teacher_weights_sha256'])
        check(language+':QAT_gradient_reaches_all_matrices',all(json.loads((student/'qat_best/gradient_checks.json').read_text()).values()))
        check(language+':actual_INT4_export_parity',json.loads((student/'qat_best/qat_export_report.json').read_text())['technical_parity_passed'])
        check(language+':full_dev_packed_parity',report['packed_vs_fake_quant_max_probability_delta']<1e-4)
        folder=student/'qat_best/onnx_int4';contract=json.loads((folder/'turn_detector_config.json').read_text())
        check(language+':own_threshold_and_codec',contract['language']==language and contract['current_truncation']=='tail' and 'language_thresholds' not in contract)
        data=Path('data_v9/monolingual_20261025')/language;rows=list(read_jsonl(data/'dev.jsonl'));values=np.load(student/'packed_dev_predictions.npz')
        labels=np.array([LABEL2ID[r['label']] for r in rows]);check(language+':dev_alignment',np.array_equal(labels,values['labels']))
        check(language+':dev_metrics_recompute',classification_metrics(labels,values['prob_EOT'],contract['threshold'])==report['overall_dev'])
        for name,metric in report['dev_slices'].items():
            mask=np.array([r['evaluation_slice']==name for r in rows]);check(language+':dev_slice:'+name,classification_metrics(labels[mask],values['prob_EOT'][mask],contract['threshold'])==metric)
        package=Path('models')/('turn_monolingual_'+language+'_20261025')/'onnx';package.mkdir(parents=True,exist_ok=False)
        for source in folder.iterdir():
            if source.is_file():shutil.copy2(source,package/source.name)
        shutil.copy2(Path('turn_v2/monolingual_standalone.py'),package/'run_model.py')
        (package/'requirements.txt').write_text('numpy\ntokenizers\nonnxruntime\n',encoding='utf-8')
        write_json(package.parent/'model_manifest.json',{'language':language,'architecture':'INDEPENDENT_MONOLINGUAL_GRU','training':'private teacher -> distillation -> INT4 QAT',
            'quality_scope':'Research only; same-language dev selection; unseen real quality not certified','teacher_sha256':report['teacher_weights_sha256'],
            'data_manifest_sha256':report['data_manifest_sha256'],'files':{p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in package.iterdir() if p.is_file()}})
        metric=report['dev_slices']['natural']
        (package.parent/'README.md').write_text('# '+language+' — model riêng, QAT INT4\n\nTeacher, vocabulary, GRU và ngưỡng riêng cho ngôn ngữ này. Research only, chưa chứng nhận90% hội thoại thực.\n\nNatural dev: accuracy '+f"{metric['accuracy']:.2%}"+', F1 EOT '+f"{metric['eot_f1']:.2%}"+'. Đây là dev dùng chọn model/ngưỡng, nhãn floor tự động.\n\nCài `python -m pip install -r onnx/requirements.txt`, chạy `python onnx/run_model.py --text "example"`. Runner dùng NumPy/Tokenizers/ONNX Runtime, không cần PyTorch.\n\nTrọng số matrix INT4, activations/GRU FP32. Văn bản hiện tại giữ token mới nhất khi vượt256token; context chỉ lấy quá khứ. Nguồn/cách chia: `README_MONOLINGUAL.md` ở workspace và manifest kèm model.\n',encoding='utf-8')
        check(language+':package_model_exact',file_hash(package/'model.onnx')==report['model_sha256'])
        check(language+':package_below112MB',sum(p.stat().st_size for p in package.iterdir() if p.is_file())<112_000_000)
        from turn_v2.monolingual_standalone import TinyOnnxDetector
        from turn_v2.monolingual_runtime import LanguageDetector
        direct=TinyOnnxDetector(package);reference=LanguageDetector(folder)
        for r in rows[:8]:
            check(language+':standalone_codec_parity:'+str(len(checks)),direct.encode(r['text'],r.get('context',''))==reference.encode(r['text'],r.get('context','')) and abs(direct.predict(r['text'],r.get('context',''))['prob_EOT']-reference.predict(r['text'],r.get('context',''))['prob_EOT'])<1e-7)
        registry['models'][language]={'path':package.as_posix(),'status':'RESEARCH_ONLY_UNSEEN_REAL_TARGET_NOT_CERTIFIED','model_sha256':file_hash(package/'model.onnx')}
    check('established_VI_unchanged',file_hash(Path(registry['models']['vi']['path'])/'model.onnx')==registry['models']['vi']['model_sha256']=='9aeae3868139d5c2faa19bee34d7128a90c41ae5b0161a8f87fc039d96188412')
    freeze=ROOT/'frozen_before_regression.json';check('regression_not_already_repeated',not freeze.exists())
    write_json(ROOT/'registry.json',registry);write_json(freeze,{'registry':registry,'registry_sha256':file_hash(ROOT/'registry.json'),
        'cohort':'Previously consumed384; regression diagnostic only, not a fresh acceptance set','threshold_selection':'Prior same-language development only'})
    router=SeparateLanguageModels(ROOT/'registry.json');rows=list(read_jsonl(Path('data_v6/pho_text_20261025/sealed.jsonl')))
    check('held_already_seen_not_fresh',Path('data_v6/pho_text_20261025/sealed_consumed.json').exists());predictions=[];metrics={}
    for language in ['vi','en','zh']:
        selected=[r for r in rows if r['language']==language];outputs=[router.predict(r['text'],r.get('context',''),language) for r in selected]
        labels=np.array([LABEL2ID[r['label']] for r in selected]);probs=np.array([o['prob_EOT'] for o in outputs]);threshold=outputs[0]['threshold']
        metrics[language]=classification_metrics(labels,probs,threshold)
        predictions.extend({'case_id':r['case_id'],'reference':r['label'],**o} for r,o in zip(selected,outputs,strict=True))
        np.savez_compressed(ROOT/(language+'_regression.npz'),labels=labels,prob_EOT=probs,case_id=np.array([r['case_id'] for r in selected]))
    write_jsonl(ROOT/'regression_predictions.jsonl',predictions)
    crawl=list(read_jsonl(Path('data_v4/fresh_crawl_20261025/review_cases.jsonl')))
    crawl=[{**r,**router.predict(r['text'],r.get('context',''),r['language']),'label':None,'is_gold':False} for r in crawl]
    write_jsonl(ROOT/'crawl_predictions.jsonl',crawl)
    player={'model':{'architecture':'THREE_INDEPENDENT_MODELS','registry':registry['models'],'accuracy':None,'eot_f1':None,'reference_labels':0},'rows':crawl}
    template=Path('tools/templates/natural_crawl_player.html').read_text(encoding='utf-8').replace('final_eval/crawl_predictions.jsonl','crawl_predictions.jsonl')
    (ROOT/'crawl_review.html').write_text(template.replace('__DATA__',json.dumps(player,ensure_ascii=False).replace('<','\\u003c')),encoding='utf-8')
    result={'status':'THREE_SEPARATE_MODELS_AVAILABLE_RESEARCH_ONLY','architecture':'Separate files, tokenizers and scalar thresholds; explicit language router',
        'regression_metrics':metrics,'regression_is_fresh_test':False,'automatic_references_only':True,'real_quality_target_certified':False,
        'crawl_cases':len(crawl),'crawl_gold_labels':0,'crawl_accuracy':None,'crawl_eot_f1':None,'VI_deployment_unchanged':True,
        'checks':checks,'all_technical_checks_passed':all(c['passed'] for c in checks)}
    write_json(ROOT/'report.json',result)
    table=''.join('<tr><td>'+language+'</td><td>'+str(m['samples'])+'</td><td>'+f"{m['accuracy']:.2%}"+'</td><td>'+f"{m['eot_f1']:.2%}"+'</td></tr>' for language,m in metrics.items())
    body='<!doctype html><html lang="vi"><meta charset="utf-8"><title>Ba model riêng</title><style>body{font:18px system-ui;max-width:1050px;margin:40px auto;padding:20px;color:#18313e}td,th{padding:12px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}p{line-height:1.7}</style><h1>Việt / Anh / Trung: mỗi tiếng một model</h1><p>Bản Việt QAT cũ giữ nguyên. Anh và Trung có teacher, GRU distill, vocabulary và QAT INT4 riêng. Bộ router chọn đúng file theo tiếng được truyền vào.</p><h2>384 mẫu regression đã xem</h2><p>Nhãn observed-floor/VAD tự động; chưa có human semantic gold. Điểm này không phải test mới và không dùng chọn lại ngưỡng/trọng số. Mỗi model dùng tokenizer/codec triển khai của mình; so sánh mô tả, chưa tách được nguyên nhân thay đổi.</p><table><tr><th>Tiếng</th><th>Mẫu</th><th>Accuracy</th><th>F1 EOT</th></tr>'+table+'</table><p>Bản Việt trước đây đạt93,55%/90,19% trên dev tiền tố cũ. Khác dữ liệu và định nghĩa tham chiếu nên không thể chuyển điểm đó thành chứng nhận cho hội thoại mới.</p><p><a href="crawl_review.html">Phát lại204prefix bằng từng model riêng</a> · <a href="report.json">JSON kết quả</a></p><h2>File model riêng</h2><pre>'+html.escape(json.dumps(registry['models'],ensure_ascii=False,indent=2))+'</pre></html>'
    (ROOT/'review.html').write_text(body,encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)


if __name__=='__main__':main()
