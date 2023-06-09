# In-Context Analogical Reasoning with Pre-Trained Language Models

Official GitHub repository for ACL 2023 paper [In-Context Analogical Reasoning with Pre-Trained Language Models](https://arxiv.org/abs/2305.17626).

## Getting Started

### Python Dependencies

```
pip install -r requirements.txt
```

### Usage

```
python task.py --config center_single --load_dir ~/lm-raven --save_dir ~/lm-raven
python images.py --config center_single --index 3044 --load_dir ~/lm-raven --save_dir ~/lm-raven
python inference.py --model_name gpt-3 --api_key sk-1234 --config center_single -b 1 -n 3 --load_dir ~/lm-raven --save_dir ~/lm-raven
python evaluation.py --path ~/lm-raven/center_single_500_gpt-3_b1_n3.json
```

## Cite

```
@inproceedings{hu2023context,
  title={In-Context Analogical Reasoning with Pre-Trained Language Models},
  author={Hu, Xiaoyang and Storks, Shane and Lewis, Richard L and Chai, Joyce},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Long Paper)},
  year={2023}
}
```
