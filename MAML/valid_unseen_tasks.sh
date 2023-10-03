MODEL='MM_Nine_tasks_FirstOrder_noScheduler' 

BASE_PATH='/home/arun'

DATA_DIR='/media/Data/MRI/datasets/unseen_artifact_suppression'

FLAG='valid_query'

VALID_TASK_STRINGS='Motion/04','Motion/06','Spatial/04','Undersampling/03','Undersampling/05','Ghosting/04','Ghosting/06','Spiking/03','Spiking/05','Noise/03','Noise/04','Gamma/01','Gamma/02','MotionM/04','MotionM/06','SpatialM/04','UndersamplingM/03','UndersamplingM/05','GhostingM/04','GhostingM/06','SpikingM/03','SpikingM/05','NoiseM/03','NoiseM/04','GammaM/01','GammaM/02'

CHECKPOINT=${BASE_PATH}'/experiments/maml/'${MODEL}'/best_model.pt'

BATCH_SIZE=1

DEVICE='cuda:0'

OUT_PATH=${BASE_PATH}'/experiments/maml/'${MODEL}'/unadapted_unseen_tasks_results'
DATA_DIR=${DATA_DIR}

echo python valid.py --checkpoint ${CHECKPOINT} --out-path ${OUT_PATH} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-dir ${DATA_DIR} --valid_task_strings ${VALID_TASK_STRINGS} --flag ${FLAG}
python valid.py --checkpoint ${CHECKPOINT} --out-path ${OUT_PATH} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-dir ${DATA_DIR} --valid_task_strings ${VALID_TASK_STRINGS} --flag ${FLAG}

