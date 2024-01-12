set -m

python3 main.py &

tritonserver --model-repository=models

fg %1
