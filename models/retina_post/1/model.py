import triton_python_backend_utils as pb_utils
import json, torch
import numpy as np
from utils import *

img_name = 'img'
loc_name = 'output'
conf_name = '781'
land_name = '780'
output_name = 'bbox'

def proc(img, loc, conf, landmarks):
    priorbox = PriorBox(image_size=(img[0].shape[0], img[0].shape[1]))
    priors = priorbox.forward()

    bboxes = detect_faces(loc, conf, landmarks, priors, img[0], conf_threshold = 0.97)
    bboxes = np.expand_dims(bboxes, axis=0).astype(np.float16)

    torch.cuda.empty_cache()
    return bboxes

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        output0_config = pb_utils.get_output_config_by_name(self.model_config, output_name)
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                # Retrieve the input tensors
                img = pb_utils.get_input_tensor_by_name(request, img_name)
                loc = pb_utils.get_input_tensor_by_name(request, loc_name)
                conf = pb_utils.get_input_tensor_by_name(request, conf_name)
                landmarks = pb_utils.get_input_tensor_by_name(request, land_name)

                if np is None:
                    error_message = "Missing required input tensor in request"
                    print(error_message)
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(error_message)
                    )
                    responses.append(inference_response)
                    continue

                img, loc, conf, landmarks = [x.as_numpy() for x in [img, loc, conf, landmarks]]
                
                out = proc(img, loc, conf, landmarks)

                out_tensor_0 = pb_utils.Tensor(output_name, out.astype(self.output0_dtype))
                inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            except Exception as e:
                error_message = f"Failed to process the request: {str(e)}"
                print(error_message)
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(error_message)
                )
            responses.append(inference_response)

        return responses
