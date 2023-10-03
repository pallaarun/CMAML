MODEL='MM_Nine_tasks_FirstOrder_noScheduler_SGD'

BASE_PATH='/home/arun'

CHECKPOINT=${BASE_PATH}'/experiments/sgd/'${MODEL}'/best_model.pt'

BATCH_SIZE=1

DEVICE='cuda:0'

TASK_STRINGS='UndersamplingM_SpikingM/3_3','NoiseM_SpikingM/3_3','SpatialM_NoiseM/2_3','GhostingM_SpikingM/3_3','UndersamplingM_GhostingM/3_3','UndersamplingM_NoiseM/3_3'

OUT_PATH=${BASE_PATH}'/experiments/sgd/'${MODEL}'/unadapted_comp_unseen_tasks_results/'

DATA_DIR='/media/Data/MRI/datasets/unseen_comp_artifact_suppression'

echo python valid.py --checkpoint ${CHECKPOINT} --out-path ${OUT_PATH} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-dir ${DATA_DIR} --task-strings ${TASK_STRINGS}

python valid.py --checkpoint ${CHECKPOINT} --out-path ${OUT_PATH} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-dir ${DATA_DIR} --task-strings ${TASK_STRINGS}