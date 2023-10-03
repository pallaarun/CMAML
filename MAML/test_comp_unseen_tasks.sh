MODEL='MM_Nine_tasks_FirstOrder_noScheduler' 

BASE_PATH='/home/arun'

EXPERIMENT_PATH=${BASE_PATH}'/experiments/maml'

CHECKPOINT=${EXPERIMENT_PATH}'/'${MODEL}'/best_model.pt'

RESULTS_DIR=${EXPERIMENT_PATH}'/'${MODEL}'/adapted_comp_unseen_tasks_results'

TENSORBOARD_SUMMARY_DIR=${EXPERIMENT_PATH}'/'${MODEL}'/adapted_comp_unseen_test_summary'

TEST_SUPPORT_BATCH_SIZE=5

DEVICE='cuda:0'

TEST_PATH='/media/Data/MRI/datasets/unseen_comp_artifact_suppression'

NUM_TEST_ADAPTATION_STEPS=50

TEST_TASK_STRINGS='UndersamplingM_SpikingM/3_3','NoiseM_SpikingM/3_3','SpatialM_NoiseM/2_3','GhostingM_SpikingM/3_3','UndersamplingM_GhostingM/3_3','UndersamplingM_NoiseM/3_3'

echo python test.py --checkpoint ${CHECKPOINT} --results_dir ${RESULTS_DIR} --tensorboard_summary_dir ${TENSORBOARD_SUMMARY_DIR} --test_support_batch_size ${TEST_SUPPORT_BATCH_SIZE} --device ${DEVICE} --test_path ${TEST_PATH} --no_of_test_adaptation_steps ${NUM_TEST_ADAPTATION_STEPS} --test_task_strings ${TEST_TASK_STRINGS}

python test.py --checkpoint ${CHECKPOINT} --results_dir ${RESULTS_DIR} --tensorboard_summary_dir ${TENSORBOARD_SUMMARY_DIR} --test_support_batch_size ${TEST_SUPPORT_BATCH_SIZE} --device ${DEVICE} --test_path ${TEST_PATH} --no_of_test_adaptation_steps ${NUM_TEST_ADAPTATION_STEPS} --test_task_strings ${TEST_TASK_STRINGS}
