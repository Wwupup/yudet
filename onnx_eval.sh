#!/usr/bin/bash

# SCORE_THRESH=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6)
# NMS_THRESH=0.45
# MODE=640,640
# MODEL_ZOO=(
# "/home/ww/projects/yudet/workspace/onnx/scrfd_500m_bnkps.onnx" \
# "/home/ww/projects/yudet/workspace/onnx/yolo5face_500m_640x640.onnx" \
# "/home/ww/projects/yudet/workspace/onnx/yunet_yunet_320_640_tinyfpn_best_simplify.onnx" \
# "/home/ww/projects/yudet/workspace/onnx/retinaface_mbnet0.25.onnx" \
# )

# for value in ${MODEL_ZOO[@]}
#     do
#     for thresh in ${SCORE_THRESH[@]}
#         do
#         echo model: $value score_tresh: $thresh 
#         python ./compare_inference.py $value --mode $MODE --nms_thresh $NMS_THRESH --score_thresh $thresh --eval 
#         done
#     done


# SCORE_THRESH=0.3
# NMS_THRESH=0.45
# MODE=640,640
# MODEL_ZOO=(
# "/home/ww/projects/yudet/workspace/onnx/scrfd_500m_bnkps.onnx" \
# "/home/ww/projects/yudet/workspace/onnx/yolo5face_500m_640x640.onnx" \
# "/home/ww/projects/yudet/workspace/onnx/yunet_yunet_320_640_tinyfpn_best_simplify.onnx" \
# "/home/ww/projects/yudet/workspace/onnx/retinaface_mbnet0.25.onnx" \
# )

# for value in ${MODEL_ZOO[@]}
#     do
#     echo model: $value score_tresh: $SCORE_THRESH 
#     python ./compare_inference.py $value --mode $MODE --nms_thresh $NMS_THRESH --score_thresh $SCORE_THRESH --eval 
#     done


SCORE_THRESH=0.3
NMS_THRESH=0.45
MODE=320,320
MODEL_ZOO=(
"./workspace/onnx/scrfd_500m_bnkps.onnx" \
"./workspace/onnx/yolo5face_500m_320x320.onnx" \
"./workspace/onnx/yunet_yunet_320_640_tinyfpn_best_simplify.onnx" \
"./workspace/onnx/retinaface_mbnet0.25.onnx" \
)

for value in ${MODEL_ZOO[@]}
    do
    echo model: $value score_tresh: $SCORE_THRESH 
    python ./compare_inference.py $value --mode $MODE --nms_thresh $NMS_THRESH --score_thresh $SCORE_THRESH --eval 
    done
