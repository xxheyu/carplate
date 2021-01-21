*************************************************************************

Enviroment

PaddlePaddle 1.8+
python3.7
glibc 2.23
CUDA 10.0
cuDNN 7.6+ (GPU)

*************************************************************************

Run Inference

conda activate paddle
python tools/infer/predict_system.py --image_dir="test_images/" \
	--det_model_dir="./inference/ch_ppocr_mobile_v1.1_det_infer/" \
	--rec_model_dir="./inference/ch_ppocr_mobile_v1.1_rec_infer/" \
	--cls_model_dir="./inference/ch_ppocr_mobile_v1.1_cls_infer/" \
	--use_angle_cls=True --use_space_char=True

The results are saved in "./inference_results" by default. 

*************************************************************************

Optional Aruguments

  --use_gpu USE_GPU
  --ir_optim IR_OPTIM
  --use_tensorrt USE_TENSORRT
  --gpu_mem GPU_MEM
  --image_dir IMAGE_DIR
  --det_algorithm DET_ALGORITHM
  --det_model_dir DET_MODEL_DIR
  --det_max_side_len DET_MAX_SIDE_LEN
  --det_db_thresh DET_DB_THRESH
  --det_db_box_thresh DET_DB_BOX_THRESH
  --det_db_unclip_ratio DET_DB_UNCLIP_RATIO
  --det_east_score_thresh DET_EAST_SCORE_THRESH
  --det_east_cover_thresh DET_EAST_COVER_THRESH
  --det_east_nms_thresh DET_EAST_NMS_THRESH
  --det_sast_score_thresh DET_SAST_SCORE_THRESH
  --det_sast_nms_thresh DET_SAST_NMS_THRESH
  --det_sast_polygon DET_SAST_POLYGON
  --rec_algorithm REC_ALGORITHM
  --rec_model_dir REC_MODEL_DIR
  --rec_image_shape REC_IMAGE_SHAPE
  --rec_char_type REC_CHAR_TYPE
  --rec_batch_num REC_BATCH_NUM
  --max_text_length MAX_TEXT_LENGTH
  --rec_char_dict_path REC_CHAR_DICT_PATH
  --use_space_char USE_SPACE_CHAR
  --vis_font_path VIS_FONT_PATH
  --use_angle_cls USE_ANGLE_CLS
  --cls_model_dir CLS_MODEL_DIR
  --cls_image_shape CLS_IMAGE_SHAPE
  --label_list LABEL_LIST
  --cls_batch_num CLS_BATCH_NUM
  --cls_thresh CLS_THRESH
  --enable_mkldnn ENABLE_MKLDNN
  --use_zero_copy_run USE_ZERO_COPY_RUN
  --use_pdserving USE_PDSERVING

*************************************************************************