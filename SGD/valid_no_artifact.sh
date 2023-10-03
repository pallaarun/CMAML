MODEL='MM_Nine_tasks_FirstOrder_noScheduler_SGD'

BASE_PATH='/home/arun'

CHECKPOINT=${BASE_PATH}'/experiments/sgd/'${MODEL}'/best_model.pt'

BATCH_SIZE=1

DEVICE='cuda:0'

TASK_STRINGS='Undersampling/03','UndersamplingM/03'

OUT_PATH=${BASE_PATH}'/experiments/sgd/'${MODEL}'/unadapted_no_artifact_results/'

DATA_DIR='/media/Data/MRI/datasets/unseen_artifact_suppression'

echo python valid_no_artifact.py --checkpoint ${CHECKPOINT} --out-path ${OUT_PATH} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-dir ${DATA_DIR} --task-strings ${TASK_STRINGS}

python valid_no_artifact.py --checkpoint ${CHECKPOINT} --out-path ${OUT_PATH} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-dir ${DATA_DIR} --task-strings ${TASK_STRINGS}