DATA_DIR='/media/Data/MRI/datasets/unseen_artifact_suppression'

FLAG='train_support'

VALID_TASK_STRINGS='Motion/04','Motion/06','Spatial/04','Undersampling/04','Undersampling/06','Ghosting/04','Ghosting/06','Spiking/04','Spiking/06','Noise/04','Noise/06','Gamma/04','Gamma/06'

DEVICE='cuda:0'

echo python data_count.py --device ${DEVICE} --data-dir ${DATA_DIR} --valid_task_strings ${VALID_TASK_STRINGS} --flag ${FLAG}
python data_count.py --device ${DEVICE} --data-dir ${DATA_DIR} --valid_task_strings ${VALID_TASK_STRINGS} --flag ${FLAG}

