MODEL='MM_Nine_tasks_FirstOrder_noScheduler' 

BASE_PATH='/home/arun/experiments/maml/'${MODEL}




                                ### ---- Needs to be changed based on the requirement ----

DATA_DIR='/media/Data/MRI/datasets/unseen_comp_artifact_suppression'

RESULTS_TYPE='adapted_comp_unseen_tasks_results'

TASK_STRINGS='UndersamplingM_SpikingM/3_3','NoiseM_SpikingM/3_3','SpatialM_NoiseM/2_3','GhostingM_SpikingM/3_3','UndersamplingM_GhostingM/3_3','UndersamplingM_NoiseM/3_3'

                                                    ### ---- END ----





REPORT_PATH=${BASE_PATH}'/'${RESULTS_TYPE}'_txt_reports'

FLAG='valid_query'

echo python evaluate.py --base-path ${BASE_PATH} --data-dir ${DATA_DIR} --report-path ${REPORT_PATH} --task_strings ${TASK_STRINGS} --flag ${FLAG} --results-type ${RESULTS_TYPE}

python evaluate.py --base-path ${BASE_PATH} --data-dir ${DATA_DIR} --report-path ${REPORT_PATH} --task_strings ${TASK_STRINGS} --flag ${FLAG} --results-type ${RESULTS_TYPE}