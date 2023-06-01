## In-Context Analogical Reasoning with Pre-Trained Language Models

## Getting Started

### Python Dependencies
```
pip install accelerate
pip install bitsandbytes
pip install numpy
pip install openai
pip install Pillow
pip install Requests
pip install torch
pip install tqdm
pip install transformers
```

### Usage
```
python3 task.py --config center_single --load_dir ~/lm-raven --save_dir ~/lm-raven
python3 images.py --config center_single --index 3044 --load_dir ~/lm-raven --save_dir ~/lm-raven
python3 inference.py --model_name gpt-3 --api_key sk-1234 --config center_single -b 1 -n 3 --load_dir ~/lm-raven --save_dir ~/lm-raven
python3 evaluation.py --path ~/lm-raven/center_single_500_gpt-3_b1_n3.json
```
