"""Consolidated inspectable report for text-only natural-conversation rounds."""
from collections import Counter
import json
from pathlib import Path
import re
import shutil

from build_notebook import cell

ROOT=Path('outputs/natural_v7_20261025')


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def row_metrics(name,metrics,scope):
    return [{'model':name,'language':lang,'scope':scope,**value} for lang,value in metrics.items()]


def main():
    ROOT.mkdir(exist_ok=True)
    v5=load(Path('outputs/natural_v5_20261025/runs/natural_v5_train_20261025/report.json'))
    small=load(Path('outputs/natural_v6_20261025/runs/natural_v6_minilm_finetune_20261025/report.json'))
    qwen=load(Path('outputs/natural_v6_20261025/qwen_heads/report.json'))
    mini=load(Path('outputs/natural_v6_20261025/minilm_heads/report.json'))
    values=[]
    values+=row_metrics('V5 mDeBERTa',v5['best_dev']['natural_languages'],'V5 dev; sampled prior context, Whisper-small')
    values+=row_metrics('V6 MiniLM fine-tune',small['best_dev']['natural_languages'],'V6 dev; current-turn/immediate-partner text, Whisper-small')
    values+=row_metrics('V6 Qwen feature head',{lang:item['dev'] for lang,item in qwen['winners'].items()},'V6 dev; frozenQwen + class/call weighted logistic head')
    values+=row_metrics('V6 MiniLM feature head',{lang:item['dev'] for lang,item in mini['winners'].items()},'V6 dev; frozenMiniLM + class/call weighted logistic head')
    v7path=ROOT/'runs/natural_v7_text_20261025/report.json'
    v7=load(v7path) if v7path.exists() else None
    if v7:values+=row_metrics('V7 mDeBERTa + PhoWhisper',v7['best_dev']['natural_languages'],'V7 dev; samecases/callgroups, repairedVIASR')
    v8path=Path('outputs/natural_v8_20261025/runs/natural_v8_lora_20261025_r3/report.json')
    v8=load(v8path) if v8path.exists() else None
    if v8:values+=row_metrics('V8 Qwen supervised LoRA',v8['best_dev']['natural_languages'],'SamePhoWhisper dev; supervisedLoRA, fulltext-only teacher')
    exact_path=ROOT/'exact_dev_calibration.json';exact=load(exact_path) if exact_path.exists() else None
    if exact:
        chosen=next(c for c in exact['candidates'] if c['path']==exact['selected_path'] and c['kind']==exact['selected_kind'])
        values+=row_metrics('Chốt ngưỡng exact dev',chosen['dev']['natural_languages'],'SamePhoWhisper dev; weights fixed, exact threshold boundaries')
    final_path=ROOT/'final_eval/report.json';final=load(final_path) if final_path.exists() else None
    asrpath=Path('data_v6/pho_text_20261025/manifest.json');asr=load(asrpath) if asrpath.exists() else None
    comparepath=ROOT/'matched_ASR_dev/report.json';compare=load(comparepath) if comparepath.exists() else None
    result={'status':('FINAL_HELD_EVALUATED_TARGET_UNMET' if not final['automatic_held_numeric_target_met'] else 'AUTOMATIC_HELD_TARGET_MET_NOT_HUMAN_CERTIFIED') if final else 'COMPLETED_DEV_CAMPAIGNS_TARGET_UNMET' if v7 else 'NEXT_ROUND_IN_PROGRESS_TARGET_UNMET',
        'objective':{'languages':['vi','en','zh'],'accuracy_strictly_above':.9,'eot_f1_strictly_above':.9,'classifier_input':'TEXT_ONLY'},
        'rows':values,'v5_stop':v5['stop_reason'],'v6_small_stop':small['stop_reason'],'v7_stop':v7['stop_reason'] if v7 else None,
        'new_natural_train':1433,'new_natural_dev':293,'held_cases':384,'held_classifier_scored':bool(final),'final_held':final,'v8_stop':v8['stop_reason'] if v8 else None,
        'human_gold_available':False,'crawl_cases':204,'crawl_gold_labels':0,'crawl_accuracy':None,'crawl_eot_f1':None,
        'ASR_diagnostics':asr['diagnostics'] if asr else None,'matched_ASR_dev':compare,
        'accepted_serving_default_changed':False,'v7_teacher':{'path':v7['best_path'],'sha256':v7['checkpoint_sha256']} if v7 else None,
        'new_quality_qualified_distilled_model_exported':False,'user_target_achieved':False}
    (ROOT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md='''# Hội thoại tự nhiên: các vòng text-only V5–V8

## Material Passport

Experiment run: **TRAINED/DEV_EVALUATED, QUALITY_TARGET_UNMET**. Mục tiêu: accuracy và F1 EOT đều >90% riêng cho Việt/Anh/Trung. Nguồn công khai; nhãn nhường lượt VAD/observed-floor tự động, chưa có human gold. Mô hình chỉ nhận text và lời đối diện đã xuất hiện trong quá khứ.

Train natural1.433mẫu từ4cuộc gọi Việt,101cuộc gọi Anh,93cuộc gọi Trung; dev293mẫu.384mẫu giữ riêng từ1cuộc gọi Việt,13cuộc gọi Anh,4cuộc gọi Trung. Các điểm cùng cuộc gọi không độc lập, nhất là Việt dev chỉ có1cuộc gọi. Replay chỉ lấy V4 train; pilot70mẫu V4 đã xem là regression. Điểm dev không phải điểm test mới.

| Model | Tiếng | Accuracy dev | F1 EOT dev | EOT nhầm/hold | Phạm vi |
| --- | --- | ---: | ---: | ---: | --- |
'''
    for row in values:
        md+=f"| {row['model']} | {row['language']} | {row['accuracy']:.2%} | {row['eot_f1']:.2%} | {row['false_eot_rate']:.2%} | {row['scope']} |\n"
    md+='''
V5: full mDeBERTa, balanced language/label sampling, replay,3vòng đã chạy1.800bước rồi dừng theo6lần dev không cải thiện. Checkpoint tốt nhất vòng2,bước300, không chọn checkpoint cuối đơn thuần.

V6 dựng lại current turn và immediate previous partner theo audio/time; text chỉ đến thời điểm quyết định. Vẫn cùng case, nhãn và call split; hai người nói của một cuộc gọi nằm trong cùng tập. Encoder feature heads học train rồi chọn regularization/ngưỡng trên dev. MiniLM cross-encoder có21,44triệu tham số học, embeddings cố định; đã train300bước, dừng dev plateau. Đây là thử nghiệm riêng, không thay bản Việt QAT mặc định.

Không coi Qwen là nhãn gold. Qwen lớn và các head nhỏ chưa vượt mục tiêu nên chưa dùng chúng để tạo một gói distill/INT4 được chấp nhận. MiniLM FP32 khoảng470,6MB có tiềm năng nén dưới112MB ở4bit, nhưng kích thước chưa phải chứng cứ chất lượng. Chưa xuất gói V6 đạt90%.

V6 ASR repetition diagnostic:37/628mẫu Việt train và6/145mẫu Việt dev có lặp3-gram trên30%. Đây không phải WER và chưa có transcript người chấm. V7 đổi tuyến ASR Việt sang PhoWhisper-large trên đúng cửa sổ đã quan sát; không bỏ mẫu theo điểm model, không sửa nhãn/call split. Anh/Trung giữ nguyên transcript V6. So sánh ASR matched-dev cố định teacher/codec/case/label/ngưỡng; tuyến ASR bao gồm model và beam decoding, chưa tách được đóng góp riêng của từng yếu tố.

'''
    if asr:md+='PhoWhisper diagnostics (không phải WER):\n\n```json\n'+json.dumps(asr['diagnostics'],ensure_ascii=False,indent=2)+'\n```\n\n'
    if v7:md+=f"V7: dừng `{v7['stop_reason']}`; checkpoint chọn bằng dev `{v7['best_path']}`, SHA256 `{v7['checkpoint_sha256']}`.\n\n"
    if v8:md+=f"V8: Qwen3-4B đã học supervised LoRA thật sự (khác frozen head ở V6), chạy{v8['steps_completed']}bước, dừng `{v8['stop_reason']}`. Adapter cần đi cùng base8GB đã khóa revision; kích thước adapter không phải kích thước triển khai. Không phải RL, chưa QAT/exportINT4.\n\n"
    if exact:md+='Sau khi cố định checkpoint, ngưỡng cuối được chọn tại ranh giới xác suất chính xác trên cùng dev; lưới0,005 trước đó có thể bỏ sót các score rất sát0,5. Không đổi trọng số, không dùng dự đoán test. Toàn bộ xác suất dev và hash được giữ trong `exact_dev_calibration.json`.\n\n'
    md+='''384mẫu giữ riêng chưa được chấm classifier trong các vòng dev này.204prefix crawl vẫn chưa có nhãn độc lập nên accuracy/F1 crawl=null. Không gọi model đạt90% từ train-fit, từ tỷ lệ EOT lớn hoặc từ điểm judge tự gán.

[Bảng kết quả](outputs/natural_v7_20261025/review.html) · [Player204prefix V4](outputs/stream_v4_20261025/review.html) · [Dữ liệu/recipe V5](README_NATURAL_V5.md). Ô cuối notebook `natural_campaign_status` đọc trạng thái các job, không khởi chạy lại chúng.

Nguồn: [DataHive CC BY4.0](https://huggingface.co/datasets/datahiveai/multilingual-conversations-sample/tree/1120cd7533b16ba8dd7a7c07e6d316b74b60f72a); [Scicom/CallHome](https://huggingface.co/datasets/Scicom-intl/semantic-vad-eot/tree/98d1445eab71e4084ccb838dbd87a205f6f2e293), CallHome giữ giấy phép nguồn CC BY-NC-SA4.0; [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554), Apache2.0; [MiniLM đa ngôn ngữ](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e8f8c211226b894fcb81acc59f3b34ba3efd5f42), Apache2.0; [PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large/tree/b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb), BSD3Clause. PhoWhisper attribution: Thanh-Thien Le, Linh The Nguyen, Dat Quoc Nguyen, ICLR2024 Tiny Papers.

Đã tìm thêm VietMed/VietMed-Sum có transcript người chấm. Chưa đưa vào train EOT: đoạn audio ASR được cắt không tự là ranh giới nhường lượt, cần gắn được timing và chuyển người nói trước. Không tự lấy cuối đoạn làm EOT.
'''
    if final:
        md=md.replace('**TRAINED/DEV_EVALUATED, QUALITY_TARGET_UNMET**','**FINAL_PUBLIC_HELD_EVALUATED, '+('AUTOMATIC_NUMERIC_TARGET_MET_NOT_HUMAN_CERTIFIED' if final['automatic_held_numeric_target_met'] else 'QUALITY_TARGET_UNMET')+'**')
        md=md.replace('[Player204prefix V4](outputs/stream_v4_20261025/review.html)','[Player 204 prefix: model đã chọn](outputs/natural_v7_20261025/crawl_review.html)')
        md=md.replace('384mẫu giữ riêng chưa được chấm classifier trong các vòng dev này.','384mẫu giữ riêng chỉ được chấm sau khi khóa lựa chọn cuối, không dùng để tune dev. Sau lượt chấm này chúng chuyển thành regression.')
        md+='\n## Kiểm tra cuối trên384mẫu giữ riêng\n\n'
        md+='Mô hình chọn trước test: `'+final['selected_kind']+'`, `'+final['selected_path']+'`. Nhãn automatic observed-floor/VAD, chưa có human gold; Việt1cuộc gọi, Anh13, Trung4. Chỉ là point estimates trên cohort nhỏ, không khẳng định chất lượng mọi hội thoại.\n\n'
        md+='| Tiếng | Mẫu | Accuracy | F1 EOT | EOT nhầm/hold |\n| --- | ---: | ---: | ---: | ---: |\n'
        for lang,item in final['metrics']['selected'].items():md+=f"| {lang} | {item['samples']} | {item['accuracy']:.2%} | {item['eot_f1']:.2%} | {item['false_eot_rate']:.2%} |\n"
        md+='\nNumeric target trên nhãn tự động: `'+str(final['automatic_held_numeric_target_met'])+'`.204prefix crawl chưa có nhãn nên accuracy/F1 vẫn=null. Không thay model serving hoặc quảng bá bản distill đạt90%.\n'
    source_audit=ROOT/'public_source_audit/ViYT-Diar_candidate.json'
    if source_audit.exists():
        audit=load(source_audit)
        md+='\nNguồn kiểm chứng Việt tiếp theo đã khảo sát: [ViYT-Diar]('+audit['source_url']+'), 100 bản ghi, tác giả công bố mốc diarization gán thủ công. Đã lưu dataset card và khóa revision, chưa tải audio/chạy classifier. Vì đây là official test, giữ riêng cho đánh giá. Mốc người nói không tự là nhãn semantic EOT: còn cần recipe handoff/pause, loại đoạn chồng lấn theo quy tắc khóa trước và ASR chỉ lấy lời đã xuất hiện. Không lấy nó làm train rồi gọi điểm trên cùng tập là test.\n'
    Path('README_NATURAL_CAMPAIGN.md').write_text(md,encoding='utf-8')
    payload=json.dumps(result,ensure_ascii=False).replace('<','\\u003c')
    page=Path('tools/templates/natural_campaign.html').read_text(encoding='utf-8').replace('__DATA__',payload)
    (ROOT/'review.html').write_text(page,encoding='utf-8')
    if final:
        crawl=[json.loads(line) for line in (ROOT/'final_eval/crawl_predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        assert len(crawl)==204 and all(r['label'] is None and r['is_gold'] is False for r in crawl)
        player={'model':{'kind':final['selected_kind'],'path':final['selected_path'],'identity_sha256':final['selected_identity'],
            'thresholds':final['thresholds'],'accuracy':None,'eot_f1':None,'reference_labels':0},'rows':crawl}
        template=Path('tools/templates/natural_crawl_player.html').read_text(encoding='utf-8')
        (ROOT/'crawl_review.html').write_text(template.replace('__DATA__',json.dumps(player,ensure_ascii=False).replace('<','\\u003c')),encoding='utf-8')
    notebook=Path('a.ipynb');nb=load(notebook)
    if not any('natural_campaign_status' in c.get('metadata',{}).get('tags',[]) for c in nb['cells']):
        backup=ROOT/'a_before_campaign_status.ipynb'
        if not backup.exists():shutil.copy2(notebook,backup)
        code='''
        import json
        from pathlib import Path
        NC_ROOT = Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
        def nc_read(relative):
            for base in [NC_ROOT, NC_ROOT/'outputs/natural_v7_20261025', NC_ROOT/'outputs/natural_v6_20261025', NC_ROOT/'outputs/natural_v5_20261025']:
                path=base/relative
                if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
            return None
        nc_jobs={}
        for name in ['natural_v5_train_20261025','natural_v6_finetune_20261025','natural_v6_pho_repair_20261025','natural_v7_compare_asr_20261025','natural_v7_text_20261025']:
            nc_jobs[name]=nc_read('jobs/'+name+'/status.json')
        print(json.dumps({'model_input':'TEXT_ONLY','jobs':nc_jobs,
            'PhoWhisper_progress':nc_read('data_v6/pho_text_20261025/progress.json'),
            'teacher_progress':nc_read('runs/natural_v7_text_20261025/progress.json'),
            'teacher_selection':nc_read('runs/natural_v7_text_20261025/selection.json'),
            'note':'Dev is not independent test. This cell only reads status; it never launches/restarts training.'},ensure_ascii=False,indent=2))
        '''
        nb['cells'].extend([cell('markdown','''
        ## V6–V7: ngữ cảnh trước đó, teacher nhỏ và sửa ASR Việt

        Giữ đầu vào text-only. Đã thử MiniLM fine-tune và Qwen feature head trên train/dev tách theo cuộc gọi; chưa đạt90%. Đang/đã xử lý lại cùng cửa sổ Việt bằng PhoWhisper-large rồi thích nghi teacher với text mới; không đổi nhãn hay loại mẫu theo điểm model.

        [Bảng kết quả và ASR review](outputs/natural_v7_20261025/review.html) · [Recipe và phạm vi kết luận](README_NATURAL_CAMPAIGN.md). Ô dưới đọc trạng thái từ kernel Colab đang kết nối; có thể chạy lại để làm mới mà không khởi chạy thêm job.
        ''','natural_campaign_status'),cell('code',code,'natural_campaign_status')])
        notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    code=next(c for c in nb['cells'] if c['cell_type']=='code' and 'natural_campaign_status' in c.get('metadata',{}).get('tags',[]))
    if final:
        markdown=next(c for c in nb['cells'] if c['cell_type']=='markdown' and 'natural_campaign_status' in c.get('metadata',{}).get('tags',[]))
        markdown['source']=cell('markdown','''
        ## V5–V8: kết quả hội thoại tự nhiên trên ba ngôn ngữ

        Đã train thêm teacher, MiniLM và Qwen3-4B LoRA với đầu vào text-only. Lựa chọn model/ngưỡng đã khóa trên dev trước khi chấm 384 mẫu giữ riêng; các mẫu đó nay là regression. Nhãn tham chiếu observed-floor/VAD tự động, chưa có human gold. Bảng dưới đọc kết quả Colab đã lưu.

        [Kết quả kiểm tra cuối](outputs/natural_v7_20261025/review.html) · [Phát lại 204 prefix crawl](outputs/natural_v7_20261025/crawl_review.html) · [Nghe 20 ví dụ ASR](outputs/natural_v7_20261025/asr_review/review.html) · [Recipe và giới hạn kết luận](README_NATURAL_CAMPAIGN.md).
        ''','natural_campaign_status')['source']
        notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    (ROOT/'notebook_status_cell.py').write_text(''.join(code['source']),encoding='utf-8')
    print(json.dumps({'rows':len(values),'notebook_cells':len(nb['cells']),'v7_finished':bool(v7),'status':result['status']}),flush=True)


if __name__=='__main__':main()
