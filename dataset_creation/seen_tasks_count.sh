DATA_DIR='/media/Data/MRI/datasets/artifact_suppression'

FLAG='train_support'

VALID_TASK_STRINGS='Motion/03','Motion/05','Motion/07','Spatial/02','Spatial/02','Spatial/02','Undersampling/03','Undersampling/05','Undersampling/07'

DEVICE='cuda:0'

echo python data_count.py --device ${DEVICE} --data-dir ${DATA_DIR} --valid_task_strings ${VALID_TASK_STRINGS} --flag ${FLAG}
python data_count.py --device ${DEVICE} --data-dir ${DATA_DIR} --valid_task_strings ${VALID_TASK_STRINGS} --flag ${FLAG}