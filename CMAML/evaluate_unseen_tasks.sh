MODEL='MM_Curriculum_Nine_tasks_FirstOrder_noScheduler' 

BASE_PATH='/home/arun/experiments/maml/'${MODEL}




                                ### ---- Needs to be changed based on the requirement ----

DATA_DIR='/media/Data/MRI/datasets/unseen_artifact_suppression'

RESULTS_TYPE='adapted_unseen_tasks_results'

TASK_STRINGS='Motion/04','Motion/06','Spatial/04','Undersampling/03','Undersampling/05','Ghosting/04','Ghosting/06','Spiking/03','Spiking/05','Noise/03','Noise/04','Gamma/01','Gamma/02','MotionM/04','MotionM/06','SpatialM/04','UndersamplingM/03','UndersamplingM/05','GhostingM/04','GhostingM/06','SpikingM/03','SpikingM/05','NoiseM/03','NoiseM/04','GammaM/01','GammaM/02'

                                                    ### ---- END ----





REPORT_PATH=${BASE_PATH}'/'${RESULTS_TYPE}'_txt_reports'

FLAG='valid_query'

echo python evaluate.py --base-path ${BASE_PATH} --data-dir ${DATA_DIR} --report-path ${REPORT_PATH} --task_strings ${TASK_STRINGS} --flag ${FLAG} --results-type ${RESULTS_TYPE}

python evaluate.py --base-path ${BASE_PATH} --data-dir ${DATA_DIR} --report-path ${REPORT_PATH} --task_strings ${TASK_STRINGS} --flag ${FLAG} --results-type ${RESULTS_TYPE}