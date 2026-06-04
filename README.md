### Installation
Need to install onnxruntime library:
1) `wget https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-1.26.0.tgz`
2) `tar -xzf onnxruntime-linux-x64-1.26.0.tgz`
3) `sudo mv onnxruntime-linux-x64-1.26.0 /opt/onnxruntime`

Note: the standard ONNX Runtime release tarball usually does **not** include the OpenVINO execution provider, so the node will fall back to CPU unless you install or build an ONNX Runtime package that was compiled with OpenVINO support.

If you want OpenVINO acceleration specifically, the runtime you link against must report the OpenVINO execution provider as available; otherwise the warning in `opi_detection_node.cpp` is expected and harmless.

At launch time, you can disable the OpenVINO attempt entirely with `enable_openvino_ep:=false` if you just want the node to run on CPU without the provider warning.

Practical suggestion: train with CUDA in Python, export to ONNX, then run the ROS node on CPU first. If you want extra inference speed later, the next thing to try is an ONNX Runtime build that includes OpenVINO support or a direct OpenVINO inference path.
