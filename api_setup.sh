#!/bin/bash

# Path to the TensorRT trtexec binary
TRTEXEC=/usr/src/tensorrt/bin/trtexec

# Path to ONNX files
retina_ONNX=/opt/tritonserver/onnx/retina.onnx

retina_MODEL=models/retina/1/model.plan

$TRTEXEC --onnx=$retina_ONNX --saveEngine=$retina_MODEL --verbose --minShapes=input:1x3x512x512 --optShapes=input:1x3x1440x1440 --maxShapes=input:1x3x1440x1440 --fp16 --useCudaGraph
sh run.sh
#python3 main.py &
