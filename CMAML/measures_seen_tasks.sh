MODEL='MM_Curriculum_Nine_tasks_FirstOrder_noScheduler' 
BASE_PATH='/home/arun'
DATA_DIR='/media/Data/MRI'

DATA_FLAG='valid_query'

                                                ### --- Needs to be changed accordingly --- ###
RESULTS_TYPE='adapted_seen_tasks_results'
                                                            ### --- END --- ###

for ONE_TASK in 'MotionM/03' 'MotionM/05' 'MotionM/07' 'SpatialM/02' 'SpatialM/03' 'SpatialM/05' 'UndersamplingM/02' 'UndersamplingM/04' 'UndersamplingM/06'
    do

    echo ${ONE_TASK}, ${RESULTS_TYPE}

    TARGET_PATH=${DATA_DIR}'/datasets/MM_artifact_suppression/'${ONE_TASK}'/'${DATA_FLAG}
    PARTIAL_PREDICTION_PATH=${BASE_PATH}'/experiments/maml/'${MODEL}'/'${RESULTS_TYPE}'/'
    REPORT_PATH=${BASE_PATH}'/experiments/maml/'${MODEL}'/csv_reports_for_'${RESULTS_TYPE}'/'

    python measures.py --partial_prediction_path ${PARTIAL_PREDICTION_PATH} --target_path ${TARGET_PATH} --report-path ${REPORT_PATH} --data_flag ${DATA_FLAG} --one_task ${ONE_TASK}

    done