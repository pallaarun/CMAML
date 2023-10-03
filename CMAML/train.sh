MODEL='1_5_11_MM_Curriculum_Nine_tasks_FirstOrder_noScheduler' 

BASE_PATH='/media/Data/MRI/datasets/MM_artifact_suppression'

EXP_PATH='/home/arun'

# DEGRADATION_NAMES='Motion/03','Motion/05','Motion/07','Motion/09','Spatial/02','Spatial/03','Spatial/04','Spatial/08','Undersampling/04','Undersampling/05','Undersampling/08'

# TRAIN_TASK_STRINGS='Motion/03','Motion/05','Motion/07','Motion/09','Spatial/02','Spatial/03','Spatial/04','Spatial/08','Undersampling/04','Undersampling/05','Undersampling/08'

EASY_TASK_STRINGS='MotionM/03','SpatialM/02','UndersamplingM/02'
MEDIUM_TASK_STRINGS='MotionM/05','SpatialM/03','UndersamplingM/04'
DIFFICULT_TASK_STRINGS='MotionM/07','SpatialM/05','UndersamplingM/06'

PACING_FUNCTION='1','5','11'

VISUALIZE_TASK_STRING='MotionM/05'

TRAIN_SUPPORT_BATCH_SIZE=5
VAL_SUPPORT_BATCH_SIZE=5

TRAIN_QUERY_BATCH_SIZE=5
VAL_QUERY_BATCH_SIZE=5

TRAIN_TASK_BATCH_SIZE=3
VAL_TASK_BATCH_SIZE=3

# TRAIN_NUM_ADAPTATION_STEPS=1
# VAL_NUM_ADAPTATION_STEPS=1
PACING_TRAIN_NUM_ADAPTATION_STEPS='1','2','3'
PACING_VAL_NUM_ADAPTATION_STEPS='1','2','3'

NUM_EPOCHS=200

DEVICE='cuda:0'


EXP_DIR=${EXP_PATH}'/experiments/maml/'${MODEL}

TRAIN_PATH=${BASE_PATH}

#VALIDATION_PATH=${BASE_PATH}'/datasets/'

echo python train.py --train_task_batch_size ${TRAIN_TASK_BATCH_SIZE} --val_task_batch_size ${VAL_TASK_BATCH_SIZE} --visualize_task_string ${VISUALIZE_TASK_STRING} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train_path ${TRAIN_PATH} --train_support_batch_size ${TRAIN_SUPPORT_BATCH_SIZE} --train_query_batch_size ${TRAIN_QUERY_BATCH_SIZE} --val_support_batch_size ${VAL_SUPPORT_BATCH_SIZE} --val_query_batch_size ${VAL_QUERY_BATCH_SIZE} --pacing_no_of_train_adaptation_steps ${PACING_TRAIN_NUM_ADAPTATION_STEPS} --pacing_no_of_val_adaptation_steps ${PACING_VAL_NUM_ADAPTATION_STEPS} --pacing_function ${PACING_FUNCTION} --easy_task_strings ${EASY_TASK_STRINGS} --medium_task_strings ${MEDIUM_TASK_STRINGS} --difficult_task_strings ${DIFFICULT_TASK_STRINGS}

python train.py --train_task_batch_size ${TRAIN_TASK_BATCH_SIZE} --val_task_batch_size ${VAL_TASK_BATCH_SIZE} --visualize_task_string ${VISUALIZE_TASK_STRING} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train_path ${TRAIN_PATH} --train_support_batch_size ${TRAIN_SUPPORT_BATCH_SIZE} --train_query_batch_size ${TRAIN_QUERY_BATCH_SIZE} --val_support_batch_size ${VAL_SUPPORT_BATCH_SIZE} --val_query_batch_size ${VAL_QUERY_BATCH_SIZE} --pacing_no_of_train_adaptation_steps ${PACING_TRAIN_NUM_ADAPTATION_STEPS} --pacing_no_of_val_adaptation_steps ${PACING_VAL_NUM_ADAPTATION_STEPS} --pacing_function ${PACING_FUNCTION} --easy_task_strings ${EASY_TASK_STRINGS} --medium_task_strings ${MEDIUM_TASK_STRINGS} --difficult_task_strings ${DIFFICULT_TASK_STRINGS}
