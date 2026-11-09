import hashlib
import json
from types import SimpleNamespace
import pytest
from tools.storage_guard import ensure_free_space,reuse_checkpoint,download_checkpoint


def test_preserve_reserve_before_new_write(tmp_path,monkeypatch):
    monkeypatch.setattr('tools.storage_guard.shutil.disk_usage',lambda p:SimpleNamespace(free=15))
    assert ensure_free_space(tmp_path,5,reserve_bytes=10)==15
    with pytest.raises(OSError):ensure_free_space(tmp_path,6,reserve_bytes=10)


def test_verified_assembled_checkpoint_does_not_need_chunks(tmp_path):
    target=tmp_path/'runs/best.safetensors';target.parent.mkdir();target.write_bytes(b'weights')
    manifest=tmp_path/'transfer.json';manifest.write_text(json.dumps({'path':'runs/best.safetensors','bytes':7,'sha256':hashlib.sha256(b'weights').hexdigest(),'parts':[]}))
    assert reuse_checkpoint(tmp_path,manifest)
    target.write_bytes(b'changed')
    with pytest.raises(ValueError):reuse_checkpoint(tmp_path,manifest)


def test_resume_rejects_wrong_range_without_overwriting_partial(tmp_path,monkeypatch):
    target=tmp_path/'model.safetensors';partial=tmp_path/'model.safetensors.partial';partial.write_bytes(b'old')
    digest=hashlib.sha256(b'oldnew').hexdigest()
    (tmp_path/'model.safetensors.partial.json').write_text(json.dumps({'remote_path':'content/model','bytes':6,'sha256':digest}))
    runtime=SimpleNamespace(base_url='https://test.invalid/',http=SimpleNamespace(get=lambda *a,**k:SimpleNamespace(status_code=200,headers={})))
    monkeypatch.setattr('tools.storage_guard.ensure_free_space',lambda *a,**k:100)
    with pytest.raises(ValueError,match='resume range'):download_checkpoint(runtime,'content/model',target,6,digest)
    assert partial.read_bytes()==b'old' and not target.exists()


def test_stream_produces_one_verified_file_without_chunk_set(tmp_path,monkeypatch):
    class Response:
        status_code=200;headers={'content-type':'application/octet-stream'}
        def iter_content(self,size):yield b'abc';yield b'def'
    runtime=SimpleNamespace(base_url='https://test.invalid/',http=SimpleNamespace(get=lambda *a,**k:Response()))
    monkeypatch.setattr('tools.storage_guard.ensure_free_space',lambda *a,**k:100)
    target=tmp_path/'model.safetensors';result=download_checkpoint(runtime,'content/model',target,6,hashlib.sha256(b'abcdef').hexdigest())
    assert target.read_bytes()==b'abcdef' and result['temporary_full_copies']==0
    assert not list(tmp_path.glob('part_*.bin')) and not (tmp_path/'model.safetensors.partial').exists()
