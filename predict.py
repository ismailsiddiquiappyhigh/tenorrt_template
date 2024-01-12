import warnings
warnings.filterwarnings('ignore')
import time, gc
import sys
import numpy as np
import tritonclient.grpc as grpcclient

try:
    keepalive_options = grpcclient.KeepAliveOptions(
        keepalive_time_ms=2**31 - 1,
        keepalive_timeout_ms=20000,
        keepalive_permit_without_calls=False,
        http2_max_pings_without_data=2
    )
    triton_client = grpcclient.InferenceServerClient(
        url='localhost:8001',
        verbose=False,
        keepalive_options=keepalive_options)
except Exception as e:
    print("channel creation failed: " + str(e))
    sys.exit()


def triton_infer(inp, model_name, input_name, output_name):
    inputs = []
    outputs = []
    
    for i, x in enumerate(input_name):
        temp = grpcclient.InferInput(x, inp[i].shape, "FP16")
        temp.set_data_from_numpy(inp[i])
        inputs.append(temp)
        
    for x in output_name:
        outputs.append(grpcclient.InferRequestedOutput(x))

    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    
    res = []
    for x in output_name:
        temp = results.as_numpy(x)
        print(x, temp.shape)
        res.append(temp)
    
    return res

def retina_det(img):
    print('Input Size', img.size)
    img = img.convert('RGB')
    img.thumbnail((1024, 1024))
    print('Resized Size', img.size)

    img = np.array(img)
    img = np.expand_dims(img, 0).astype('float16')

    srt = time.time()
    res = triton_infer([img], 'ensemble_model', ['ensemble_input'], ['ensemble_output'])

    print('Infer Time: ', time.time()-srt)
    res = res[0].tolist()
    gc.collect()
    return res