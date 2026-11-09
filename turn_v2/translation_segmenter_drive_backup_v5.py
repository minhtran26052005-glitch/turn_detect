"""Full-epoch backup: support both initial raw-only and trained-generation restore."""
from . import translation_segmenter_drive_backup_v2 as previous
import hashlib
import zipfile


class Backup(previous.Backup):
    def zip_source(self):
        launcher=previous.old.read_json(previous.old.ROOT/'launcher.json')
        super().zip_source()
        module=launcher['module'];relative='turn_v2/'+module.split('.')[-1]+'.py'
        with zipfile.ZipFile(previous.old.JOB/'source_bundle.zip') as archive:
            checksum=hashlib.sha256(archive.read(relative)).hexdigest()
        assert checksum==launcher['source_sha256'],'Active coordinator code differs from source bundle'
        self.active_source_proof={'module':module,'source_sha256':checksum}

    def manifest(self,complete):
        result=super().manifest(complete)
        result['required_restore_for_full_epoch']='turn_v2.restore_full_epoch_campaign_v4'
        path=previous.old.ROOT/'launcher.json'
        if path.exists():
            result['active_campaign_launcher']=previous.old.read_json(path)
            if complete:
                proof=self.active_source_proof;active=result['active_campaign_launcher']
                assert proof['module']==active['module'] and proof['source_sha256']==active['source_sha256'],'Coordinator changed during backup; preserve partial and retry'
                result['verified_active_campaign_source']=proof
        return result


if __name__=='__main__':
    previous.Backup=Backup
    previous.main()
