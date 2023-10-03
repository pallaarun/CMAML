MODEL='MM_Curriculum_Nine_tasks_FirstOrder_noScheduler' 

BASE_PATH='/home/arun'

EXPERIMENT_PATH=${BASE_PATH}'/experiments/maml'

CHECKPOINT=${EXPERIMENT_PATH}'/'${MODEL}'/best_model.pt'

RESULTS_DIR=${EXPERIMENT_PATH}'/'${MODEL}'/adapted_unseen_tasks_results'

TENSORBOARD_SUMMARY_DIR=${EXPERIMENT_PATH}'/'${MODEL}'/adapted_unseen_test_summary'

TEST_SUPPORT_BATCH_SIZE=5

DEVICE='cuda:0'

TEST_PATH='/media/Data/MRI/datasets/unseen_artifact_suppression'

NUM_TEST_ADAPTATION_STEPS=50

TEST_TASK_STRINGS='Motion/04','Motion/06','Spatial/04','Undersampling/03','Undersampling/05','Ghosting/04','Ghosting/06','Spiking/03','Spiking/05','Noise/03','Noise/04','Gamma/01','Gamma/02','MotionM/04','MotionM/06','SpatialM/04','UndersamplingM/03','UndersamplingM/05','GhostingM/04','GhostingM/06','SpikingM/03','SpikingM/05','NoiseM/03','NoiseM/04','GammaM/01','GammaM/02'

echo python test.py --checkpoint ${CHECKPOINT} --results_dir ${RESULTS_DIR} --tensorboard_summary_dir ${TENSORBOARD_SUMMARY_DIR} --test_support_batch_size ${TEST_SUPPORT_BATCH_SIZE} --device ${DEVICE} --test_path ${TEST_PATH} --no_of_test_adaptation_steps ${NUM_TEST_ADAPTATION_STEPS} --test_task_strings ${TEST_TASK_STRINGS}

python test.py --checkpoint ${CHECKPOINT} --results_dir ${RESULTS_DIR} --tensorboard_summary_dir ${TENSORBOARD_SUMMARY_DIR} --test_support_batch_size ${TEST_SUPPORT_BATCH_SIZE} --device ${DEVICE} --test_path ${TEST_PATH} --no_of_test_adaptation_steps ${NUM_TEST_ADAPTATION_STEPS} --test_task_strings ${TEST_TASK_STRINGS}










