"""Advance export metadata after the completed packed full-dev evaluation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import file_hash,write_json


def main():
    root=Path('outputs/monolingual_20261025');receipt=root/'release_metadata_receipt.json';assert not receipt.exists()
    parent=root/'completion_validation.json';assert json.loads(parent.read_text())['status']=='PASS'
    original=root/'deployment_archives.json';shutil.copy2(original,root/'deployment_archives_before_metadata_fix.json');items=[]
    for language in ['en','zh']:
        package=Path('models')/('turn_monolingual_'+language+'_20261025');folder=package/'onnx';path=folder/'turn_detector_config.json'
        report=json.loads((root/f'runs/mono_{language}_student_20261025/monolingual_report.json').read_text())
        assert file_hash(folder/'model.onnx')==report['model_sha256'] and not report['natural_dev_target_met']
        contract=json.loads(path.read_text());assert contract['quality_status']=='PACKED_QAT_EXPORT_PENDING_FULL_DEV_EVALUATION'
        write_json(root/(language+'_contract_before_full_dev_metadata.json'),contract)
        contract.update(quality_status='FULL_DEV_EVALUATED_TARGET_UNMET_RESEARCH_ONLY',quality_scope='Same-language packed ONNX dev evaluated; automatic floor/proxy labels; unseen real quality not certified')
        write_json(path,contract)
        manifest_path=package/'model_manifest.json';manifest=json.loads(manifest_path.read_text());manifest['files']={p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in folder.iterdir() if p.is_file()}
        manifest['metadata_finalizer_sha256']=file_hash(Path(__file__));write_json(manifest_path,manifest)
        archive=package.with_suffix('.zip');previous=package.with_name(package.name+'_before_metadata_fix').with_suffix('.zip');assert not previous.exists();archive.rename(previous)
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(package.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(package.parent))
        completed=subprocess.run([sys.executable,str(folder/'run_model.py'),'--text','test'],capture_output=True,text=True);assert completed.returncode==0,completed.stderr
        output=json.loads(completed.stdout);assert output['language']==language and output['quality_status']==contract['quality_status']
        assert file_hash(folder/'model.onnx')==report['model_sha256']
        items.append({'language':language,'zip':str(archive),'zip_bytes':archive.stat().st_size,'zip_sha256':file_hash(archive),'model_bytes':(folder/'model.onnx').stat().st_size,
            'package_bytes':sum(p.stat().st_size for p in package.rglob('*') if p.is_file()),'standalone_CLI':'PASS','quality_status':contract['quality_status']})
    write_json(original,items);write_json(receipt,{'status':'PASS','weights_and_thresholds_unchanged':True,'parent_validation_sha256':file_hash(parent),
        'source_sha256':file_hash(Path(__file__)),'archives':items,'note':'Export pending label advanced after verified full-dev evaluation; prior artifacts preserved'})
    print(json.dumps(items),flush=True)


if __name__=='__main__':main()
