MODEL='MM_Nine_tasks_FirstOrder_noScheduler_SGD'

BASE_PATH='/home/arun'

CHECKPOINT=${BASE_PATH}'/experiments/sgd/'${MODEL}'/best_model.pt'

BATCH_SIZE=1

DEVICE='cuda:0'

TASK_STRINGS='MotionM/03','MotionM/05','MotionM/07','SpatialM/02','SpatialM/03','SpatialM/05','UndersamplingM/02','UndersamplingM/04','UndersamplingM/06'

OUT_PATH=${BASE_PATH}'/experiments/sgd/'${MODEL}'/unadapted_seen_tasks_results/'

DATA_DIR='/media/Data/MRI/datasets/MM_artifact_suppression'

echo python valid.py --checkpoint ${CHECKPOINT} --out-path ${OUT_PATH} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-dir ${DATA_DIR} --task-strings ${TASK_STRINGS}

python valid.py --checkpoint ${CHECKPOINT} --out-path ${OUT_PATH} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-dir ${DATA_DIR} --task-strings ${TASK_STRINGS}