MODEL='MM_Nine_tasks_FirstOrder_noScheduler' 

BASE_PATH='/media/Data/MRI/datasets/MM_artifact_suppression'

EXP_PATH='/home/arun'

DEGRADATION_NAMES='MotionM/03','MotionM/05','MotionM/07','SpatialM/02','SpatialM/03','SpatialM/05','UndersamplingM/02','UndersamplingM/04','UndersamplingM/06'

TRAIN_TASK_STRINGS='MotionM/03','MotionM/05','MotionM/07','SpatialM/02','SpatialM/03','SpatialM/05','UndersamplingM/02','UndersamplingM/04','UndersamplingM/06'

VISUALIZE_TASK_STRING='MotionM/05'

TRAIN_SUPPORT_BATCH_SIZE=5
VAL_SUPPORT_BATCH_SIZE=5

TRAIN_QUERY_BATCH_SIZE=5
VAL_QUERY_BATCH_SIZE=5

TRAIN_TASK_BATCH_SIZE=3
VAL_TASK_BATCH_SIZE=3

TRAIN_NUM_ADAPTATION_STEPS=1
VAL_NUM_ADAPTATION_STEPS=1

NUM_EPOCHS=200

DEVICE='cuda:0'

EXP_DIR=${EXP_PATH}'/experiments/maml/'${MODEL}

TRAIN_PATH=${BASE_PATH}

echo python train.py --train_task_batch_size ${TRAIN_TASK_BATCH_SIZE} --val_task_batch_size ${VAL_TASK_BATCH_SIZE} --task_strings ${TRAIN_TASK_STRINGS} --degradation_names ${DEGRADATION_NAMES} --visualize_task_string ${VISUALIZE_TASK_STRING} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train_path ${TRAIN_PATH} --train_support_batch_size ${TRAIN_SUPPORT_BATCH_SIZE} --train_query_batch_size ${TRAIN_QUERY_BATCH_SIZE} --val_support_batch_size ${VAL_SUPPORT_BATCH_SIZE} --val_query_batch_size ${VAL_QUERY_BATCH_SIZE} --no_of_train_adaptation_steps ${TRAIN_NUM_ADAPTATION_STEPS} --no_of_val_adaptation_steps ${VAL_NUM_ADAPTATION_STEPS}

python train.py --train_task_batch_size ${TRAIN_TASK_BATCH_SIZE} --val_task_batch_size ${VAL_TASK_BATCH_SIZE} --task_strings ${TRAIN_TASK_STRINGS} --degradation_names ${DEGRADATION_NAMES} --visualize_task_string ${VISUALIZE_TASK_STRING} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train_path ${TRAIN_PATH} --train_support_batch_size ${TRAIN_SUPPORT_BATCH_SIZE} --train_query_batch_size ${TRAIN_QUERY_BATCH_SIZE} --val_support_batch_size ${VAL_SUPPORT_BATCH_SIZE} --val_query_batch_size ${VAL_QUERY_BATCH_SIZE} --no_of_train_adaptation_steps ${TRAIN_NUM_ADAPTATION_STEPS} --no_of_val_adaptation_steps ${VAL_NUM_ADAPTATION_STEPS}
