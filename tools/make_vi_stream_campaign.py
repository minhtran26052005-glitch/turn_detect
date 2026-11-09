"""Create a Vietnamese campaign using per-batch SQLite token reads."""
from pathlib import Path

source = Path('tools/translation_segmenter_vi_only_campaign.py').read_text(encoding='utf-8')
old = """command = [sys.executable, '-u', '-m', 'turn_v2.translation_segmenter_full_train_v5',
                       '--language', 'vi', '--generation', manifest['generation'],
                       '--epoch', str(epoch), '--budget', str(train.CHUNK_STEPS)]"""
new = """command = [sys.executable, '-u', '-m', 'turn_v2.translation_segmenter_vi_stream_v1',
                       '--generation', manifest['generation'], '--epoch', str(epoch),
                       '--budget', str(train.CHUNK_STEPS)]"""
assert old in source
source = source.replace(old, new, 1).replace("'module': 'turn_v2.translation_segmenter_full_train_v5'",
                                           "'module': 'turn_v2.translation_segmenter_vi_stream_v1'")
source = source.replace("'campaign': 'translation_segmenter_vi_only_v1'",
                        "'campaign': 'translation_segmenter_vi_stream_v1'")
target = Path('tools/translation_segmenter_vi_stream_campaign.py')
target.write_text(source, encoding='utf-8')
compile(source, str(target), 'exec')
