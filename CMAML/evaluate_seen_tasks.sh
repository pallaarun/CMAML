MODEL='MM_Curriculum_Nine_tasks_FirstOrder_noScheduler'

BASE_PATH='/home/arun/experiments/maml/'${MODEL}




                                ### ---- Needs to be changed based on the requirement ----

DATA_DIR='/media/Data/MRI/datasets/MM_artifact_suppression'

RESULTS_TYPE='adapted_seen_tasks_results'

TASK_STRINGS='MotionM/03','MotionM/05','MotionM/07','SpatialM/02','SpatialM/03','SpatialM/05','UndersamplingM/02','UndersamplingM/04','UndersamplingM/06'

                                                    ### ---- END ----





REPORT_PATH=${BASE_PATH}'/'${RESULTS_TYPE}'_txt_reports'

FLAG='valid_query'

echo python evaluate.py --base-path ${BASE_PATH} --data-dir ${DATA_DIR} --report-path ${REPORT_PATH} --task_strings ${TASK_STRINGS} --flag ${FLAG} --results-type ${RESULTS_TYPE}

python evaluate.py --base-path ${BASE_PATH} --data-dir ${DATA_DIR} --report-path ${REPORT_PATH} --task_strings ${TASK_STRINGS} --flag ${FLAG} --results-type ${RESULTS_TYPE}