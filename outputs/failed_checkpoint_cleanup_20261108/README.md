# User-authorized failed checkpoint cleanup — 08/11/2026

Removed77SSH checkpoint/export files from completed rejected under20M, TinyGRU(student/candidates/fixtures only, teacher kept), ASR-only and effective16 trials. Protected v7paused fullresume/best/pretrained, historicalcontrols/teacher, frozenTESTpins outside those rejected trials and all datasets/source/reports/traces. User authorization supersedes prior retain-rejected-weights instructions for these exact files.

Removed16Windows files with explicit failed-gate evidence: three DEV_TARGET_NOT_MET GRU runs, stream_v4 user_target_achieved=false checkpoints, and specific int8_quint8_no_constfold candidate dev_quality_target_met=false. Research-only/pending-held-evaluation/accepted models were not inferred to be failed and were retained.

No recursive directory deletion. Every target absolute path checked within project and not symlink, size/mtime/inode(SSH) checked against inventory, SHA256 recorded before deletion, absence verified. All unselected inventoried Windows models still exist with original size; all protected SSH files present. Mainpaused fullcheckpointSHA unchanged feb4bc2e6cbd4d8c7fa3148f49edb86500c898045a33cc1db8a334bcc03255a8.

Measuredfree-space gain: SSH3.2198GiB, Windows1.3126GiB. LogicalsumSSH3.3531GiB differs because hardlinks share physicalstorage. WindowsCaftercleanup about12.57GiB. Completed rejected branches are intentionally no longer weight-replayable from removed files; JSONreports/SHA/protocol/traces preserved. No claim of Drivebackup. No TRAIN restarted.

See deletion_plan.json, deletion_receipt.json, remote_inventory.json, windows_inventory.json. Remote side has per-file hashed deletion receipt; each affected branch has a removal tombstone.
