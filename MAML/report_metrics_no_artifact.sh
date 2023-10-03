MODEL='MM_Nine_tasks_FirstOrder_noScheduler' 

echo "\n"
echo ${MODEL}
echo "\n"

BASE_PATH='/home/arun'

EXPERIMENT_PATH=${BASE_PATH}'/experiments/maml'

BASE_REPORT_PATH=${EXPERIMENT_PATH}'/'${MODEL}


                                    ### ---- Needs to be changed based on the requirement ----




REULTS_TYPE='adapted_no_artifact_results_txt_reports'

for one_task in  'Undersampling_amount_03' 'UndersamplingM_amount_03'


                                                    ### ---- END ----


    do

    REPORT_PATH=${BASE_REPORT_PATH}'/'${REULTS_TYPE}'/report_'${one_task}'.txt'
    
    echo ${REPORT_PATH}
    cat ${REPORT_PATH}
    echo "\n"
    
    done