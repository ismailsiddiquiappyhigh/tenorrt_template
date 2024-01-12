import triton_python_backend_utils as pb_utils
import json
import numpy as np

def preprocess(image):
    
    image_data = np.array(image)

    image_data = image_data.astype(np.float16)
    image_data -= (104, 117, 123)
    image_data = image_data.transpose(2, 0, 1)

    image_data = np.expand_dims(image_data, 0)
    return image_data
    
input_name = 'input_raw'
output_name = 'pre_out'

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        output0_config = pb_utils.get_output_config_by_name(self.model_config, output_name)
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])

    def execute(self, requests):
        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, input_name)
            if in_0 is None:
                error_message = "Input tensor 'INPUT_raw' not found in request"
                print(error_message)
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(error_message)
                )
                responses.append(inference_response)
                continue

            try:
                img = in_0.as_numpy()
                img = np.squeeze(img)
                if img is None:
                    raise ValueError("Processed image is None")
                img = preprocess(img)
                if img is None:
                    raise ValueError("Processed image is None")

                out_tensor_1 = pb_utils.Tensor(output_name, img.astype(self.output0_dtype))
                inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_1])
            except Exception as e:
                error_message = f"Failed to process the request: {str(e)}"
                print(error_message)
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(error_message)
                )
            responses.append(inference_response)

        return responses
