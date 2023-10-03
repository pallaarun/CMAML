MODEL='MM_Nine_tasks_FirstOrder_noScheduler' 

echo "\n"
echo ${MODEL}
echo "\n"

BASE_PATH='/home/arun'

EXPERIMENT_PATH=${BASE_PATH}'/experiments/maml'

BASE_REPORT_PATH=${EXPERIMENT_PATH}'/'${MODEL}


                                    ### ---- Needs to be changed based on the requirement ----





REULTS_TYPE='adapted_unseen_tasks_results_txt_reports'

# for one_task in  'MotionM_amount_03' 'MotionM_amount_05' 'MotionM_amount_07' 'SpatialM_amount_02' 'SpatialM_amount_03' 'SpatialM_amount_05' 'UndersamplingM_amount_02' 'UndersamplingM_amount_04' 'UndersamplingM_amount_06'
for one_task in  'Motion_amount_04' 'Motion_amount_06' 'Spatial_amount_04' 'Undersampling_amount_03' 'Undersampling_amount_05' 'Ghosting_amount_04' 'Ghosting_amount_06' 'Spiking_amount_03' 'Spiking_amount_05' 'Noise_amount_03' 'Noise_amount_04' 'Gamma_amount_01' 'Gamma_amount_02' 'MotionM_amount_04' 'MotionM_amount_06' 'SpatialM_amount_04' 'UndersamplingM_amount_03' 'UndersamplingM_amount_05' 'GhostingM_amount_04' 'GhostingM_amount_06' 'SpikingM_amount_03' 'SpikingM_amount_05' 'NoiseM_amount_03' 'NoiseM_amount_04' 'GammaM_amount_01' 'GammaM_amount_02'


                                                    ### ---- END ----


    do

    REPORT_PATH=${BASE_REPORT_PATH}'/'${REULTS_TYPE}'/report_'${one_task}'.txt'
    
    echo ${REPORT_PATH}
    cat ${REPORT_PATH}
    echo "\n"
    
    done