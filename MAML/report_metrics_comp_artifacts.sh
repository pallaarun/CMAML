MODEL='MM_Nine_tasks_FirstOrder_noScheduler' 

echo "\n"
echo ${MODEL}
echo "\n"

BASE_PATH='/home/arun'

EXPERIMENT_PATH=${BASE_PATH}'/experiments/maml'

BASE_REPORT_PATH=${EXPERIMENT_PATH}'/'${MODEL}


                                    ### ---- Needs to be changed based on the requirement ----





REULTS_TYPE='adapted_comp_unseen_tasks_results_txt_reports'

# for one_task in  'MotionM_amount_03' 'MotionM_amount_05' 'MotionM_amount_07' 'SpatialM_amount_02' 'SpatialM_amount_03' 'SpatialM_amount_05' 'UndersamplingM_amount_02' 'UndersamplingM_amount_04' 'UndersamplingM_amount_06'
for one_task in  'UndersamplingM_SpikingM_amount_3_3' 'NoiseM_SpikingM_amount_3_3' 'SpatialM_NoiseM_amount_2_3' 'GhostingM_SpikingM_amount_3_3' 'UndersamplingM_GhostingM_amount_3_3' 'UndersamplingM_NoiseM_amount_3_3'


                                                    ### ---- END ----


    do

    REPORT_PATH=${BASE_REPORT_PATH}'/'${REULTS_TYPE}'/report_'${one_task}'.txt'
    
    echo ${REPORT_PATH}
    cat ${REPORT_PATH}
    echo "\n"
    
    done