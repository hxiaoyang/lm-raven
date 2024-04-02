# In-Context Analogical Reasoning with Pre-Trained Language Models
Official GitHub repository for ACL 2023 paper [In-Context Analogical Reasoning with Pre-Trained Language Models](https://aclanthology.org/2023.acl-long.109/).

## Getting Started

### Datasets
* Download RAVEN-10000 [here](https://drive.google.com/file/d/111swnEzAY2NfZgeyAhVwQujMjRUfeyuY/view).
* Follow instructions [here](https://github.com/husheng12345/SRAN) to generate the I-RAVEN dataset.

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
  * If you wish to experiment with another random subset of RAVEN, replace `subset.json`.
* `evaluation.py` evaluates model performance.

## Examples
```
$ python task.py --config center_single --load_dir ~/lm-raven --save_dir ~/lm-raven
$ python images.py --config center_single --index 3044 --load_dir ~/lm-raven --save_dir ~/lm-raven
$ python inference.py --model_name gpt-3 --api_key sk-1234 --config center_single -b 1 -n 3 --load_dir ~/lm-raven --save_dir ~/lm-raven
$ python evaluation.py --path ~/lm-raven/center_single_500_gpt-3_b1_n3.json
```

## Citation
```
@inproceedings{hu-etal-2023-context,
    title = "In-Context Analogical Reasoning with Pre-Trained Language Models",
    author = "Hu, Xiaoyang  and
      Storks, Shane  and
      Lewis, Richard  and
      Chai, Joyce",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.109",
    doi = "10.18653/v1/2023.acl-long.109",
    pages = "1953--1969",
    abstract = "Analogical reasoning is a fundamental capacity of human cognition that allows us to reason abstractly about novel situations by relating them to past experiences. While it is thought to be essential for robust reasoning in AI systems, conventional approaches require significant training and/or hard-coding of domain knowledge to be applied to benchmark tasks. Inspired by cognitive science research that has found connections between human language and analogy-making, we explore the use of intuitive language-based abstractions to support analogy in AI systems. Specifically, we apply large pre-trained language models (PLMs) to visual Raven{'}s Progressive Matrices (RPM), a common relational reasoning test. By simply encoding the perceptual features of the problem into language form, we find that PLMs exhibit a striking capacity for zero-shot relational reasoning, exceeding human performance and nearing supervised vision-based methods. We explore different encodings that vary the level of abstraction over task features, finding that higher-level abstractions further strengthen PLMs{'} analogical reasoning. Our detailed analysis reveals insights on the role of model complexity, in-context learning, and prior knowledge in solving RPM tasks.",
}
```
