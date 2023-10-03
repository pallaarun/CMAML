MODEL='MM_Nine_tasks_FirstOrder_noScheduler_SGD'

BASE_PATH='/home/arun'

EXPERIMENT_PATH=${BASE_PATH}'/experiments/sgd'

CHECKPOINT=${EXPERIMENT_PATH}'/'${MODEL}'/best_model.pt'

RESULTS_DIR=${EXPERIMENT_PATH}'/'${MODEL}'/adapted_no_artifact_results'

TENSORBOARD_SUMMARY_DIR=${EXPERIMENT_PATH}'/'${MODEL}'/adapted_no_artifact_summary'

TEST_SUPPORT_BATCH_SIZE=5

DEVICE='cuda:0'

TEST_PATH='/media/Data/MRI/datasets/unseen_artifact_suppression'

NUM_TEST_ADAPTATION_STEPS=50

TEST_TASK_STRINGS='Undersampling/03','UndersamplingM/03'

echo python test_no_artifact.py --checkpoint ${CHECKPOINT} --results_dir ${RESULTS_DIR} --tensorboard_summary_dir ${TENSORBOARD_SUMMARY_DIR} --test_support_batch_size ${TEST_SUPPORT_BATCH_SIZE} --device ${DEVICE} --test_path ${TEST_PATH} --no_of_test_adaptation_steps ${NUM_TEST_ADAPTATION_STEPS} --test_task_strings ${TEST_TASK_STRINGS}

python test_no_artifact.py --checkpoint ${CHECKPOINT} --results_dir ${RESULTS_DIR} --tensorboard_summary_dir ${TENSORBOARD_SUMMARY_DIR} --test_support_batch_size ${TEST_SUPPORT_BATCH_SIZE} --device ${DEVICE} --test_path ${TEST_PATH} --no_of_test_adaptation_steps ${NUM_TEST_ADAPTATION_STEPS} --test_task_strings ${TEST_TASK_STRINGS}
