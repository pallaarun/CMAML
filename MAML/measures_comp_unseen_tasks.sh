MODEL='MM_Nine_tasks_FirstOrder_noScheduler' 
BASE_PATH='/home/arun'
DATA_DIR='/media/Data/MRI'

DATA_FLAG='valid_query'

                                                ### --- Needs to be changed accordingly --- ###
RESULTS_TYPE='adapted_comp_unseen_tasks_results'
                                                            ### --- END --- ###

for ONE_TASK in 'UndersamplingM_SpikingM/3_3' 'NoiseM_SpikingM/3_3' 'SpatialM_NoiseM/2_3' 'GhostingM_SpikingM/3_3' 'UndersamplingM_GhostingM/3_3' 'UndersamplingM_NoiseM/3_3'
    do

    echo ${ONE_TASK}, ${RESULTS_TYPE}

    TARGET_PATH=${DATA_DIR}'/datasets/unseen_comp_artifact_suppression/'${ONE_TASK}'/'${DATA_FLAG}
    PARTIAL_PREDICTION_PATH=${BASE_PATH}'/experiments/maml/'${MODEL}'/'${RESULTS_TYPE}'/'
    REPORT_PATH=${BASE_PATH}'/experiments/maml/'${MODEL}'/csv_reports_for_'${RESULTS_TYPE}'/'

    python measures.py --partial_prediction_path ${PARTIAL_PREDICTION_PATH} --target_path ${TARGET_PATH} --report-path ${REPORT_PATH} --data_flag ${DATA_FLAG} --one_task ${ONE_TASK}

    done