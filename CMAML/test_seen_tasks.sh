MODEL='MM_Curriculum_Nine_tasks_FirstOrder_noScheduler' 

BASE_PATH='/home/arun'

EXPERIMENT_PATH=${BASE_PATH}'/experiments/maml'

CHECKPOINT=${EXPERIMENT_PATH}'/'${MODEL}'/best_model.pt'

RESULTS_DIR=${EXPERIMENT_PATH}'/'${MODEL}'/adapted_seen_tasks_results'

TENSORBOARD_SUMMARY_DIR=${EXPERIMENT_PATH}'/'${MODEL}'/adapted_seen_test_summary'

TEST_SUPPORT_BATCH_SIZE=5

DEVICE='cuda:0'

TEST_PATH='/media/Data/MRI/datasets/MM_artifact_suppression'

NUM_TEST_ADAPTATION_STEPS=50

TEST_TASK_STRINGS='MotionM/03','MotionM/05','MotionM/07','SpatialM/02','SpatialM/03','SpatialM/05','UndersamplingM/02','UndersamplingM/04','UndersamplingM/06'

echo python test.py --checkpoint ${CHECKPOINT} --results_dir ${RESULTS_DIR} --tensorboard_summary_dir ${TENSORBOARD_SUMMARY_DIR} --test_support_batch_size ${TEST_SUPPORT_BATCH_SIZE} --device ${DEVICE} --test_path ${TEST_PATH} --no_of_test_adaptation_steps ${NUM_TEST_ADAPTATION_STEPS} --test_task_strings ${TEST_TASK_STRINGS}

python test.py --checkpoint ${CHECKPOINT} --results_dir ${RESULTS_DIR} --tensorboard_summary_dir ${TENSORBOARD_SUMMARY_DIR} --test_support_batch_size ${TEST_SUPPORT_BATCH_SIZE} --device ${DEVICE} --test_path ${TEST_PATH} --no_of_test_adaptation_steps ${NUM_TEST_ADAPTATION_STEPS} --test_task_strings ${TEST_TASK_STRINGS}










