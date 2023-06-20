# In-Context Analogical Reasoning with Pre-Trained Language Models
Official GitHub repository for ACL 2023 paper [In-Context Analogical Reasoning with Pre-Trained Language Models](https://arxiv.org/abs/2305.17626).

## Getting Started

### Datasets
* Download RAVEN-10000 [here](https://drive.google.com/file/d/111swnEzAY2NfZgeyAhVwQujMjRUfeyuY/view); learn more [here](http://wellyzhang.github.io/project/raven.html).
* Follow instructions [here](https://github.com/husheng12345/SRAN) to generate the I-RAVEN dataset; learn more [here](https://arxiv.org/abs/2002.06838).

### Python Dependencies
```
$ pip install -r requirements.txt
```

## Usage
* `task.py` transcribes task data.
* Optional: `images.py` visualizes a given RPM.
* `inference.py` applies specified language model on transcribed data.
  * `gpt-3` requires an `api_key`.
  * `opt` models (e.g., `opt-125m`, `opt-1.3b`, `opt-13b` etc.) require sufficient GPU memory.
  * Argument `b = 0,1` controls whether or not to branch over components and attributes.
  * Argument `n = 1,2,3` controls the number of rows to include.
  * If you wish to experiment with a different subset of RAVEN, replace `subset.json`.
* `evaluation.py` evaluates model performance.
* For example:
```
$ python task.py --config center_single --load_dir ~/lm-raven --save_dir ~/lm-raven
$ python images.py --config center_single --index 3044 --load_dir ~/lm-raven --save_dir ~/lm-raven
$ python inference.py --model_name gpt-3 --api_key sk-1234 --config center_single -b 1 -n 3 --load_dir ~/lm-raven --save_dir ~/lm-raven
$ python evaluation.py --path ~/lm-raven/center_single_500_gpt-3_b1_n3.json
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
