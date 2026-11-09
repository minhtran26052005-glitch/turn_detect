"""Source-verified full-epoch backup with the complete recovery router."""
from . import translation_segmenter_drive_backup_v5 as previous
from . import translation_segmenter_drive_backup_v2 as driver


class Backup(previous.Backup):
    def manifest(self,complete):
        result=super().manifest(complete)
        result['required_restore_for_full_epoch']='turn_v2.restore_full_epoch_campaign_v6'
        return result


if __name__=='__main__':
    driver.Backup=Backup
    driver.main()
