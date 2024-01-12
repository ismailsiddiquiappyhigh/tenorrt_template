import warnings
warnings.filterwarnings('ignore')
import io, subprocess
import base64, time
import traceback
from PIL import Image
from typing import List
from json import dumps
from pydantic import BaseModel, Field
from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from predict import retina_det

app = FastAPI()

class Input(BaseModel):
    image_str: str

def pil_to_b64(img):
    im_file = io.BytesIO()
    img.save(im_file, format="webp")
    im_bytes = im_file.getvalue()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_body = await request.json()
    print('Error on Request:')
    print(request_body)
    print('---'*25)
    for error in exc.errors():
        for key in ['type', 'loc', 'msg']:
            print(f'{key} : {error.get(key)}')
            if error.get(key) == 'missing':
                return Response(content=dumps({"detail": exc.errors(), "body": exc.body}, default=str),
                status_code=421,
                headers={"Content-Type": "application/json"})

    print('---'*25)
    return Response(content=dumps({"detail": exc.errors(), "body": exc.body}, default=str),
        status_code=422,
        headers={"Content-Type": "application/json"})

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_body = await request.json()
    print('HTTP Error on Request:')
    print(request_body)
    print('---'*25)
    return Response(content=dumps({"detail": str(exc.detail)}, default=str),
        status_code=exc.status_code,
        headers={"Content-Type": "application/json"})

@app.get("/healthcheck")
def healthcheck():
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0:
        gpu = True
    return Response(content=dumps({"state": "healthy", "gpu": gpu}, default=str), headers={"Content-Type": "application/json"})

@app.post("/")
def generator(input: Input):
    try:
        user_image = input.image_str
        user_img = Image.open(io.BytesIO(base64.b64decode(user_image.encode()))).convert('RGB')
        srt = time.time()
        bboxes = retina_det(user_img)
        print('Total Time: ', time.time()-srt)
        response_ = {"output": bboxes, "status_code" : 200}
    
    except Exception as e:
        request_body = dumps(input.dict(), default=str)
        print('Error on Request:')
        print(request_body)
        print('---'*25)
        response_ = "Internal Server Error: " + str(e)
        traceback_message = traceback.format_exc()
        print(traceback_message)
        return Response(content=dumps(response_, default=str), headers={"Content-Type": "application/json"}, status_code=500)
    
    return Response(content=dumps(response_, default=str), headers={"Content-Type": "application/json"})
