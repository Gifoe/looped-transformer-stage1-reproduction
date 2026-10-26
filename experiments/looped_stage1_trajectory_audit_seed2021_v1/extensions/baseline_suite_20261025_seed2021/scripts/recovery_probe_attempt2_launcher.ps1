$root='D:\looped_stage1_trajectory_audit_seed2021_v1\extensions\baseline_suite_20261025_seed2021'
& 'E:\Anaconda\python.exe' -u -X utf8 ($root+'\scripts\recovery_probe_controller.py') *> ($root+'\logs\recovery_probe_controller.log')
exit $LASTEXITCODE
