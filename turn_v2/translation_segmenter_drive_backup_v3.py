"""Same full-data backup, with explicit verified-restore entrypoint metadata."""
from . import translation_segmenter_drive_backup_v2 as previous


class Backup(previous.Backup):
    def manifest(self,complete):
        result=super().manifest(complete)
        result['required_restore_for_full_epoch']='turn_v2.restore_full_epoch_campaign_v2'
        path=previous.old.ROOT/'launcher.json'
        if path.exists():result['active_campaign_launcher']=previous.old.read_json(path)
        return result


if __name__=='__main__':
    previous.Backup=Backup
    previous.main()
