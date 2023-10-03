MODEL='MM_Nine_tasks_FirstOrder_noScheduler_SGD' 
BASE_PATH='/home/arun'
DATA_DIR='/media/Data/MRI'

DATA_FLAG='valid_query'

                                                ### --- Needs to be changed accordingly --- ###
RESULTS_TYPE='adapted_no_artifact_results'
                                                            ### --- END --- ###

for ONE_TASK in 'Undersampling/03' 'UndersamplingM/03'
    do

    echo ${ONE_TASK}, ${RESULTS_TYPE}

    TARGET_PATH=${DATA_DIR}'/datasets/unseen_artifact_suppression/'${ONE_TASK}'/'${DATA_FLAG}
    PARTIAL_PREDICTION_PATH=${BASE_PATH}'/experiments/sgd/'${MODEL}'/'${RESULTS_TYPE}'/'
    REPORT_PATH=${BASE_PATH}'/experiments/sgd/'${MODEL}'/csv_reports_for_'${RESULTS_TYPE}'/'

    python measures.py --partial_prediction_path ${PARTIAL_PREDICTION_PATH} --target_path ${TARGET_PATH} --report-path ${REPORT_PATH} --data_flag ${DATA_FLAG} --one_task ${ONE_TASK}

    done