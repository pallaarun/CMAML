MODEL='MM_Nine_tasks_FirstOrder_noScheduler' 

BASE_PATH='/home/arun'

DATA_DIR='/media/Data/MRI/datasets/unseen_comp_artifact_suppression'

FLAG='valid_query'

VALID_TASK_STRINGS='UndersamplingM_SpikingM/3_3','NoiseM_SpikingM/3_3','SpatialM_NoiseM/2_3','GhostingM_SpikingM/3_3','UndersamplingM_GhostingM/3_3','UndersamplingM_NoiseM/3_3'

CHECKPOINT=${BASE_PATH}'/experiments/maml/'${MODEL}'/best_model.pt'

BATCH_SIZE=1

DEVICE='cuda:0'

OUT_PATH=${BASE_PATH}'/experiments/maml/'${MODEL}'/unadapted_comp_unseen_tasks_results'
DATA_DIR=${DATA_DIR}

echo python valid.py --checkpoint ${CHECKPOINT} --out-path ${OUT_PATH} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-dir ${DATA_DIR} --valid_task_strings ${VALID_TASK_STRINGS} --flag ${FLAG}
python valid.py --checkpoint ${CHECKPOINT} --out-path ${OUT_PATH} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-dir ${DATA_DIR} --valid_task_strings ${VALID_TASK_STRINGS} --flag ${FLAG}

