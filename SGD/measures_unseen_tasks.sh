MODEL='MM_Nine_tasks_FirstOrder_noScheduler_SGD' 
BASE_PATH='/home/arun'
DATA_DIR='/media/Data/MRI'

DATA_FLAG='valid_query'

                                                ### --- Needs to be changed accordingly --- ###
RESULTS_TYPE='adapted_unseen_tasks_results'
                                                            ### --- END --- ###

for ONE_TASK in 'Motion/04' 'Motion/06' 'Spatial/04' 'Undersampling/03' 'Undersampling/05' 'Ghosting/04' 'Ghosting/06' 'Spiking/03' 'Spiking/05' 'Noise/03' 'Noise/04' 'Gamma/01' 'Gamma/02' 'MotionM/04' 'MotionM/06' 'SpatialM/04' 'UndersamplingM/03' 'UndersamplingM/05' 'GhostingM/04' 'GhostingM/06' 'SpikingM/03' 'SpikingM/05' 'NoiseM/03' 'NoiseM/04' 'GammaM/01' 'GammaM/02'
    do

    echo ${ONE_TASK}, ${RESULTS_TYPE}

    TARGET_PATH=${DATA_DIR}'/datasets/unseen_artifact_suppression/'${ONE_TASK}'/'${DATA_FLAG}
    PARTIAL_PREDICTION_PATH=${BASE_PATH}'/experiments/sgd/'${MODEL}'/'${RESULTS_TYPE}'/'
    REPORT_PATH=${BASE_PATH}'/experiments/sgd/'${MODEL}'/csv_reports_for_'${RESULTS_TYPE}'/'

    python measures.py --partial_prediction_path ${PARTIAL_PREDICTION_PATH} --target_path ${TARGET_PATH} --report-path ${REPORT_PATH} --data_flag ${DATA_FLAG} --one_task ${ONE_TASK}

    done