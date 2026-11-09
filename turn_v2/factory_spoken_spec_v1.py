"""Spoken-register factory transcripts, word-form numbers, separate VI/EN/ZH."""
from collections import Counter
import hashlib
import random
import re

from factory_paragraphs_v4 import normalize, make_examples as punctuation_examples
from factory_spoken_topics_v1 import (
    VI_TOPICS, EN_TOPICS, INDUSTRIES as OLD_INDUSTRIES,
    EQUIPMENT as OLD_EQUIPMENT, ROLES as OLD_ROLES, STYLES as OLD_STYLES,
)

LANGS=('vi','en','zh')
SYSTEM={
 'vi':('Sinh bản chép lời NÓI TỰ NHIÊN hư cấu bằng tiếng Việt, giữa chuyên gia/kỹ sư đến nhà máy và người phụ trách tại xưởng. '
       'Đây là lời người thật có thể nói và được ASR chép lại, không phải văn viết, biên bản, báo cáo hay bài tư vấn trau chuốt. '
       'Bắt đầu giữa lúc đang trao đổi, không chào hỏi, giới thiệu hay cảm ơn theo khuôn. Đúng bốn lượt A, B, A, B. '
       'Mỗi lượt là một đoạn nói dài, ba đến sáu câu liên quan, khoảng bốn mươi đến tám mươi từ. '
       'Câu có nhịp nói tự nhiên, mệnh đề nối tiếp; có thể hỏi lại, xác nhận, ngập ngừng nhẹ hoặc tự sửa một chi tiết khi phù hợp. '
       'Không nhét từ đệm vào mọi câu. Tránh các câu liệt kê hành chính và các đoạn giải thích như sách giáo khoa. '
       'Giữ nhất quán xưng hô giữa tôi và anh/chị, vai chuyên gia với người phụ trách. Mỗi người nói trọn đoạn rồi người kia đáp. '
       'MỌI số, số thập phân, đơn vị và mã phải viết đúng thành lời đọc, không dùng chữ số hoặc ký hiệu. '
       'Ví dụ ba phẩy hai mi li mét, một trăm hai mươi chiếc, ca chiều; dùng tên đầy đủ thay chữ viết tắt. '
       'Tự sửa lời bằng dấu phẩy, ví dụ khoảng ba, à không, gần bốn phút; không dùng dấu ba chấm hoặc ghi chú sân khấu. '
       'Dấu chấm chỉ ngắt câu đã hoàn chỉnh, dấu phẩy cho đoạn ý còn tiếp. Câu hỏi dùng dấu hỏi tự nhiên. '
       'Giữ số liệu hư cấu nhất quán, không thêm dữ kiện chưa được xác minh hay hướng dẫn thao tác nguy hiểm. '
       'Chỉ trả bốn dòng A: đoạn nói / B: đoạn nói / A: đoạn nói / B: đoạn nói. Không JSON, markdown, tiêu đề, bản dịch, chú thích hay chỉ dẫn giọng nói.'),
 'en':('Generate a FICTIONAL transcript of spontaneous SPOKEN ENGLISH between a visiting factory engineer or consultant '
       'and the responsible person on the factory floor. It must sound like something people actually say and ASR transcribes, '
       'not a written report, polished advisory answer, formal minutes, lecture or email. Start midway through the discussion; '
       'no greetings, introductions or formulaic thanks. Exactly four turns A, B, A, B. Each turn is a substantial spoken '
       'paragraph of three to six connected sentences, roughly forty to eighty words. Use natural contractions, linked clauses, '
       'practical questions, confirmations and occasional light hesitation or a self-correction when the situation calls for it. '
       'Do not put fillers in every sentence or make all turns follow the same pattern. One person finishes a paragraph before '
       'the other replies. ALL numbers, decimals, units and identifiers must be written as pronounced words, with no digit '
       'numerals or unit symbols: three point two millimeters, one hundred twenty pieces, the afternoon shift. Use full spoken '
       'names instead of written acronyms. Use commas for hesitations and repairs, not ellipses or stage directions. '
       'Full stops mark completed sentences; internal commas leave the clause continuing. Natural questions may use question '
       'marks. Keep fictional quantities consistent and distinguish an observation from a guess. No real personal details, '
       'contact information, hazardous operating instructions or translation. Output only four lines A: paragraph / B: paragraph '
       '/ A: paragraph / B: paragraph. No JSON, markdown, headings, explanations or voice annotations.'),
 'zh':('生成一段虚构的工厂现场口语转写。只用自然的普通话，像两个人实际说话后由语音识别转成文字，'
       '不要写成报告、会议纪要、书面建议、客服问答或讲课。一方是到厂的顾问或工程师，另一方是现场负责人。'
       '直接从正在讨论的事情开始，不要问候、介绍身份或套路化感谢。严格四轮，依次为A、B、A、B。'
       '每轮是一整段较长的口述，三到六个相关句子，约七十到一百六十个汉字，不要每人只说一句短话。'
       '用自然的接话、追问、确认和连贯分句，情境需要时可有少量犹豫或自我纠正，不要每句都加“嗯”“那个”。'
       '一方说完一段，对方再回应，双方称呼和职责保持一致。所有数字、小数、单位、编号都要写成读出来的汉字，'
       '例如三点二毫米、一百二十件、下午那一班，不要阿拉伯数字、单位符号或外语缩写。'
       '自我纠正用逗号连接，例如大概三，不对，应该快四分钟；不要省略号、舞台提示或括号动作。'
       '完整句子才用句号，意思还在继续时用逗号，真正提问可用问号。数量前后一致，不把推测说成已核实事实。'
       '不要真实个人信息、联系方式、危险操作步骤、翻译或外语解说。只输出四行：A：口述段落、B：口述段落、'
       'A：口述段落、B：口述段落。不要标题、编号、JSON、代码块或额外说明。')}

ZH_TOPICS=['表面缺陷与抽样办法','工序节拍差异与在制品排队','来料检验记录','批次追溯与隔离库存','维修停机安排',
 '核实振动现象与观察记录','订单进度与瓶颈工序','图纸与实际零件的尺寸差异','技术移交与人员培训','现场检查后缺少的证据',
 '物料变更与文件版本','两个班次的质量差异','试生产与验收条件','包装标签与批次标记','车间设备布局','责任分工与回复时间',
 '测量结果与量具校准','产量数据与等待时间','核实操作人员反映的问题','实际产能与计划的差异','备件协调与检查安排',
 '安全检查记录与有权限的人员','工艺变更试验的批次范围','传感器稳定性','需要复检的产品数量','尚未达成一致的讨论事项',
 '仓库条件与物料周转','首件样品与日常生产的差异','补充照片记录与测量结果','整改后再次确认的安排']
INDUSTRIES={**OLD_INDUSTRIES,'zh':['电子零件','精密机械加工','包装材料','汽车零件','纺织','家用电器','包装食品','电气设备','工程塑料','机械制造','电缆','照明设备']}
EQUIPMENT={**OLD_EQUIPMENT,'zh':['装配线','机加工工位','输送带','质量检验区','包装机','尺寸测量工位','来料区','搬运机器人','注塑机','电气检验工位','半成品仓库','表面处理工序']}
ROLES={**OLD_ROLES,'zh':[('质量顾问','生产主管'),('咨询工程师','产线监督员'),('客户工程师','质量主管'),('设备专家','维修主管'),('项目监督员','工厂负责人'),('工艺工程师','生产组长'),('技术移交专家','工厂工程师'),('客户代表','质量监督员')]}
STYLES={**OLD_STYLES,'zh':['一方不同意并要求更多证据','澄清双方理解不同的地方','对照报表和现场观察','确认待核实的事情和负责人','在设备旁边讨论刚发现的现象','看过样品或记录后继续交流']}
DELIVERY={
 'vi':['nói rõ ràng, bình thường, không cố chèn ngập ngừng','có một lần tự sửa lời hợp lý trong cả cuộc trao đổi','có một chỗ hỏi lại rồi giải thích lại bằng cách nói khác','có ít ngập ngừng và lặp nhẹ tự nhiên, vẫn giữ mạch ý'],
 'en':['ordinary clear spoken language without forced disfluencies','one plausible self-correction across the conversation','one clarification question followed by a natural rephrasing','a little natural hesitation or repetition without losing the point'],
 'zh':['普通清楚的现场口语，不硬加犹豫词','整段交流中有一次合理的自我纠正','有一次追问并换个自然说法解释','有少量自然犹豫或轻微重复，但意思连贯']}


def spoken_integer(lang,n):
    if lang=='en':
        low=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
        if n<20:return low[n]
        if n<100:return ['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety'][n//10]+(' '+low[n%10] if n%10 else '')
        return low[n//100]+' hundred'+(' '+spoken_integer(lang,n%100) if n%100 else '')
    if lang=='vi':
        low=['không','một','hai','ba','bốn','năm','sáu','bảy','tám','chín']
        if n<10:return low[n]
        if n<20:return 'mười'+(' '+('lăm' if n%10==5 else low[n%10]) if n%10 else '')
        if n<100:return low[n//10]+' mươi'+(' '+('mốt' if n%10==1 else 'lăm' if n%10==5 else low[n%10]) if n%10 else '')
        return low[n//100]+' trăm'+((' lẻ '+low[n%100]) if 0<n%100<10 else ' '+spoken_integer(lang,n%100) if n%100 else '')
    chars='零一二三四五六七八九'
    if n<10:return chars[n]
    if n<100:return (chars[n//10] if n>=20 else '')+'十'+(chars[n%10] if n%10 else '')
    return chars[n//100]+'百'+('零'+chars[n%100] if 0<n%100<10 else spoken_integer(lang,n%100) if n%100 else '')


def task_for(lang,cycle,attempt):
    seed=int.from_bytes(hashlib.sha256(f'spoken-v9:{lang}:{cycle}:{attempt}'.encode()).digest()[:8],'big');rng=random.Random(seed)
    topic=rng.randrange(30);roles=rng.choice(ROLES[lang]);units=rng.choice([80,120,160,200,240,320,500,800]);issues=rng.randrange(2,15)
    r={'id':f'{lang}_spoken_r{cycle:05d}_a{attempt:09d}','language':lang,'cycle':cycle,'attempt':attempt,'seed':seed,
       'industry':rng.choice(INDUSTRIES[lang]),'equipment':rng.choice(EQUIPMENT[lang]),'topic':{'vi':VI_TOPICS,'en':EN_TOPICS,'zh':ZH_TOPICS}[lang][topic],
       'roles':roles,'style':rng.choice(STYLES[lang]),'delivery_profile':rng.choice(DELIVERY[lang]),'batch_units':units,'observed_issues':issues,'register':'spoken_transcript'}
    a,b=spoken_integer(lang,units),spoken_integer(lang,issues)
    if lang=='vi':prompt=f"Ngành {r['industry']}, tại {r['equipment']}. Chủ đề {r['topic']}. Vai {roles[0]} và {roles[1]}. Tình huống {r['style']}. Cách nói {r['delivery_profile']}. Có thể nhắc lô {a} sản phẩm và {b} trường hợp cần xem lại, không bắt buộc dùng cả hai. Viết đúng dạng lời nói đã chép lại, số và đơn vị thành lời đọc, chỉ tiếng Việt."
    elif lang=='en':prompt=f"Industry: {r['industry']}, location: {r['equipment']}. Topic: {r['topic']}. Roles: {roles[0]} and {roles[1]}. Situation: {r['style']}. Delivery: {r['delivery_profile']}. You may mention a batch of {a} units and {b} observations to check, but need not use both. Spoken transcript only, all numbers and units as pronounced words, English only."
    else:prompt=f"行业：{r['industry']}，地点：{r['equipment']}。话题：{r['topic']}。双方角色：{roles[0]}和{roles[1]}。情境：{r['style']}。说话方式：{r['delivery_profile']}。可以提到一批{a}件产品和{b}处需要核实的情况，不必都用。只写普通话现场口语转写，所有数字和单位写成读出来的汉字。"
    r['prompt']=prompt;return r


def parse_generated(raw,task):
    markers=list(re.finditer(r'(?m)^\s*([AB])\s*[:：]\s*',raw))
    if [m.group(1) for m in markers]!=['A','B','A','B'] or raw[:markers[0].start()].strip():raise ValueError('expected_four_spoken_turns')
    turns=[]
    for i,m in enumerate(markers):
        end=markers[i+1].start() if i+1<len(markers) else len(raw);text=re.sub(r'\s+',' ',raw[m.end():end]).strip()
        if not text.endswith(('.','?','!','。','？','！')):raise ValueError('truncated_spoken_paragraph')
        if '...' in text or '…' in text:raise ValueError('ambiguous_ellipsis_boundary')
        sentences=[s.strip() for s in re.split(r'(?<=[.!?。！？])\s*',text) if s.strip()]
        if not 2<=len(sentences)<=8:raise ValueError('expected_two_to_eight_sentences')
        values=[]
        for s in sentences:
            clean=normalize(s)
            if not clean or len(s)>600 or re.search(r'[<>\[\]{}]',s):raise ValueError('invalid_spoken_sentence')
            values.append({'display':s,'normalized':clean})
        turns.append({'speaker':task['roles'][0 if m.group(1)=='A' else 1],'sentences':values})
    return {'dialogue_id':task['id'],'language':task['language'],'turns':turns,'source':'QWEN_SYNTHETIC',
            'register':'spoken_transcript','numbers_as_spoken_words':True,'is_human_gold':False,'is_audio_transcription':False}


def quality_check(dialogue,lang):
    sentences=[s for t in dialogue['turns'] for s in t['sentences']];text=' '.join(s['display'] for s in sentences);clean=normalize(text)
    if re.search(r'\d',text):raise ValueError('digit_numeral_not_spoken_word')
    if re.search(r'(?i)\b(?:mm|cm|kg|kw|mpa|rpm|plc|sku|qc|sop)\b',text) or re.search(r'\b[A-Z]{2,}\b',text):raise ValueError('written_acronym_or_unit')
    if re.search(r'[%=+*/]|\S+@\S+|https?://',text):raise ValueError('written_symbol_or_link')
    if re.search(r'[\[\]{}()]|（|）',text):raise ValueError('stage_direction_or_parenthesis')
    greetings=re.compile(r'^(?:xin chào|chào |hello\b|hi\b|good morning|good afternoon|你好|您好|大家好)',re.I)
    if any(greetings.search(s['normalized']) for s in sentences):raise ValueError('greeting_instead_of_mid_discussion')
    han=sum('\u3400'<=c<='\u9fff' for c in clean);letters=sum(c.isalpha() for c in clean)
    if lang=='zh' and (han<150 or han/max(1,letters)<.90):raise ValueError('Chinese_language_check_failed')
    if lang in ('vi','en') and han:raise ValueError('unexpected_Han_characters')
    vi_chars=sum(c in 'ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ' for c in clean)
    if lang=='vi' and vi_chars<12:raise ValueError('Vietnamese_language_check_failed')
    if lang=='en' and (vi_chars>2 or sum(w in {'the','and','we','this','that','is','to','it','our','you'} for w in clean.split())<8):raise ValueError('English_language_check_failed')
    for turn in dialogue['turns']:
        paragraph=' '.join(s['normalized'] for s in turn['sentences'])
        if lang=='zh':
            if sum('\u3400'<=c<='\u9fff' for c in paragraph)<35:raise ValueError('spoken_Chinese_turn_too_short')
        elif len(paragraph.split())<(25 if lang=='vi' else 22):raise ValueError('spoken_turn_too_short')
    if len(set(s['normalized'] for s in sentences))<len(sentences)-2:raise ValueError('repetitive_spoken_template')
    report_terms=['căn cứ vào các nội dung nêu trên','trân trọng đề nghị','the aforementioned','hereby','综上所述','特此报告','鉴于上述']
    if sum(clean.count(term) for term in report_terms)>=2:raise ValueError('written_report_register')
    dialogue['spoken_quality_checks']={'no_digit_numerals':True,'no_written_units_or_acronyms':True,'long_alternating_turns':True,
        'spoken_style_is_prompted_not_human_verified':True,'requires_human_review':True}


def make_examples(dialogue):
    rows=punctuation_examples(dialogue)
    for row in rows:row.update(register='spoken_transcript',source_is_synthetic=True)
    return rows


def advance(state):
    current=state['language']
    if state['cycle_counts'][current]>=10000:
        i=LANGS.index(current)
        if i+1<len(LANGS):state['language']=LANGS[i+1]
        else:state['cycle']+=1;state['language']=LANGS[0];state['cycle_counts']={l:0 for l in LANGS}
    return state
