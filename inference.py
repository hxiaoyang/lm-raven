import accelerate
import argparse
import json
import math
import openai
import os
import random
import re
import requests
import sys
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class Shape:
    def __init__(self, shape_dict, add_angle=False):
        self.type = self._type(shape_dict["Type"])
        self.size = self._size(shape_dict["Size"])
        self.color = self._color(shape_dict["Color"])
        if add_angle:
            self.angle = self._angle(shape_dict["Angle"])
        self.add_angle = add_angle

    def _type(self, x):
        return int(x) + 2
    
    def _size(self, x):
        return int(x) + 1

    def _color(self, x):
        return int(x)
    
    def _angle(self, x):
        return int(x)

    def __str__(self):
        if self.add_angle:
            return "({},{},{},{})".format(self.type, self.size/10,
                                          self.color*10, self.angle*100)
        else:
            return "({},{},{})".format(self.type, self.size/10, self.color*10)


class Grid:
    def __init__(self, grid_dict, dim, add_angle=False):
        self.dim = dim
        self.add_angle = add_angle
        self.coords = self._coords(grid_dict["positions"])
        self.shapes = self._shapes(grid_dict["entities"])
        self.types, self.sizes, self.colors, self.angles = self._split()
        self.string = ""
        self.layout = []
        self._update()
    
    def _coords(self, coords):
        ret = []
        for coord in coords:
            x = int(math.ceil(coord[0]*self.dim))
            y = int(math.ceil(coord[1]*self.dim))
            ret.append((x,y))
        return ret

    def _shapes(self, shape_dicts):
        return [Shape(shape_dict, add_angle=self.add_angle) for shape_dict in shape_dicts]

    def _split(self):
        types, sizes, colors, angles = [], [], [], []
        for shape in self.shapes:
            types.append(shape.type)
            sizes.append(shape.size)
            colors.append(shape.color)
            if self.add_angle:
                angles.append(shape.angle)
        types = list(set(types))
        types.sort()
        sizes = list(set(sizes))
        sizes.sort()
        colors = list(set(colors))
        colors.sort()
        angles = list(set(angles))
        angles.sort()
        return types, sizes, colors, angles
    
    def _update(self):
        self.string += "["
        for i in range(self.dim**2):
            x = int(i/self.dim) + 1
            y = i%self.dim + 1
            if (x,y) in self.coords:
                j = self.coords.index((x,y))
                self.string += str(self.shapes[j])
                self.layout.append(1)
            else:
                self.string += "-"
                self.layout.append(0)
            if i < self.dim**2 - 1:
                self.string += ", "
        self.string += "]"
        return
    
    def __str__(self):
        return self.string

    def get_layout(self):
        return str(self.layout)
    
    def get_number(self):
        return sum(self.layout)
    
    def get_types(self):
        return str(self.types)

    def get_sizes(self):
        return str(self.sizes)

    def get_colors(self):
        return str(self.colors)
    
    def get_angles(self):
        return str(self.angles)


class Branch:
    def __init__(self, arr, n=3):
        self.context = self._context(arr, n)
        self.choices = [str(x) for x in arr[8:]]

    def _context(self, arr, n):
        if n == 1:
            tpl = "{}, {}, "
            return tpl.format(*arr[6:8])
        elif n == 2:
            tpl = "row 1: {}, {}, {}; row 2: {}, {}, "
            return tpl.format(*arr[3:8])
        else:
            tpl = "row 1: {}, {}, {}; row 2: {}, {}, {}; row 3: {}, {}, "
            return tpl.format(*arr[:8])


class Component:
    def __init__(self, item_dicts, config, n=3, add_angle=False):
        self.config = config
        self.add_angle = add_angle
        self.items = self._items(item_dicts)
        self.branches = {}
        self._update(n)

    def _items(self, item_dicts):
        if self.config == "center_single":
            return [Shape(item_dict, add_angle=self.add_angle) for item_dict in item_dicts]
        elif self.config == "distribute_four":
            return [Grid(item_dict,2,add_angle=self.add_angle) for item_dict in item_dicts]
        else:
            return [Grid(item_dict,3,add_angle=self.add_angle) for item_dict in item_dicts]

    def _update(self, n):
        if self.config == "center_single":
            self.branches["type"] = [shape.type for shape in self.items]
            self.branches["size"] = [shape.size for shape in self.items]
            self.branches["color"] = [shape.color for shape in self.items]
            if self.add_angle:
                self.branches["angle"] = [shape.angle for shape in self.items]
        else:
            self.branches["type"] = [grid.get_types() for grid in self.items]
            self.branches["size"] = [grid.get_sizes() for grid in self.items]
            self.branches["color"] = [grid.get_colors() for grid in self.items]
            if self.add_angle:
                self.branches["angle"] = [grid.get_angles() for grid in self.items]
            self.branches["layout"] = [grid.get_layout() for grid in self.items]
            self.branches["number"] = [grid.get_number() for grid in self.items]
        self.branches["master"] = self.items
        for k in self.branches.keys():
            self.branches[k] = Branch(self.branches[k], n=n)


class RPM:
    def __init__(self, sample, config, n=3, add_angle=False):
        self.config = config
        self.sample = sample
        self.add_angle = add_angle
        self.components = self._components(n)
        self.context = None
        self.choices = None
        self._update(n)

    def _components(self, n):
        item_dicts_0 = [self.sample["rpm"][j][0] for j in range(16)]
        if self.config == "center_single" or self.config[:10] == "distribute":
            return [Component(item_dicts_0, self.config, n=n, add_angle=self.add_angle)]
        else:
            item_dicts_1 = [self.sample["rpm"][j][1] for j in range(16)]
            if self.config == "in_distribute_four_out_center_single":
                return [Component(item_dicts_0, "center_single", n=n, add_angle=self.add_angle),
                        Component(item_dicts_1, "distribute_four", n=n, add_angle=self.add_angle)]
            else:
                return [Component(item_dicts_0, "center_single", n=n, add_angle=self.add_angle),
                        Component(item_dicts_1, "center_single", n=n, add_angle=self.add_angle)]

    def _update(self, n):
        if self.config == "center_single" or self.config[:10] == "distribute":
            self.context = self.components[0].branches["master"].context
            self.choices = self.components[0].branches["master"].choices
        else:
            combined = []
            for x,y in zip(self.components[0].items,
                           self.components[1].items):
                combined.append("A {} / B {}".format(x,y))
            if n == 1:
                tpl = "{}, {}, "
                self.context = tpl.format(*combined[6:8])
            elif n == 2:
                tpl = "row 1: {}, {}, {}; row 2: {}, {}, "
                self.context = tpl.format(*combined[3:8])
            else:
                tpl = "row 1: {}, {}, {}; row 2: {}, {}, {}; row 3: {}, {}, "
                self.context = tpl.format(*combined[:8])
            self.choices = combined[8:]
            return


class Solver:
    def __init__(self, model_name, model=None, tokenizer=None, prefix=""):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.output = {}
        self.prefix = prefix
        self.context = None
        self.choice_scores = {}

    def __call__(self, config, load_dir, save_dir, b=1, n=3, add_angle=False, subset_dir="subset.json", experiment_name="", print_prompt=False, gpt_combine_choices=False):
        with open("{}/{}.json".format(load_dir,config), "r") as f:
            samples = json.load(f)
        subset = json.load(open(load_dir+subset_dir, 'r'))
        if gpt_combine_choices:
            self.gpt_complete_combine_choices(subset, samples, config, n=n, add_angle=add_angle, print_prompt=print_prompt)
            return
    
        for i in tqdm(subset[:500]):
            # if i % 10 < 8:
            #     continue
            sample = samples[str(i)]
            if b:
                self.output[i] = self._split(sample, config, n=n, add_angle=add_angle, print_prompt=print_prompt)
            else:
                self.output[i] = self._merge(sample, config, n=n, add_angle=add_angle, print_prompt=print_prompt)
        if experiment_name:
            file_name = f"{len(subset)}_{experiment_name}.json"
        else:
            file_name = "{}/{}_500_{}_b{}_n{}.json".format(save_dir, config,
                                                       self.model_name, b, n)
        if self.model_name != "null":
            json.dump(self.output, open(file_name, 'w'), indent=1)
            self.output, self.context = {}, None
        return

    def _split(self, sample, config, n=3, add_angle=False, print_prompt=False):
        ret = []
        rpm = RPM(sample, config, n=n, add_angle=add_angle)
        for i, component in enumerate(rpm.components):
            if self.model_name == "null":
                print(sample["rules"][i])
            ret.append({})
            for j, branch in component.branches.items():
                ret[i][j] = []
                self.context = self.prefix + branch.context
                for choice in branch.choices:
                    prompt = self.context + choice
                    if n != 1:
                        prompt += ";"
                    if self.model_name == "null" or print_prompt:
                        print(prompt)
                    if choice in self.choice_scores.keys():
                        scores = self.choice_scores[choice]
                    else:
                        if self.model_name[:3] == "gpt":
                            scores = self._gpt(prompt)
                        if self.model_name[:3] == "opt":
                            scores = self._opt(prompt)
                        if self.model_name == "mistral":
                            scores = self._mistral(prompt)
                        if self.model_name == "null":
                            scores = 0
                    ret[i][j].append(scores)
                    self.choice_scores[choice] = scores
                self.choice_scores = {}
        return ret

    def _merge(self, sample, config, n=3, add_angle=False,  print_prompt=False):
        ret = []
        rpm = RPM(sample, config, n=n, add_angle=add_angle)
        if self.model_name == "null":
            print(sample["rules"])
        self.context = self.prefix + rpm.context
        for choice in rpm.choices:
            prompt = self.context + choice
            if n != 1:
                prompt += ";"
            if self.model_name == "null" or print_prompt:
                print(prompt)
            if self.model_name[:3] == "gpt":
                scores = self._gpt(prompt)
            if self.model_name[:3] == "opt":
                scores = self._opt(prompt)
            if self.model_name == "mistral":
                scores = self._mistral(prompt)
            if self.model_name == "null":
                scores = 0
            ret.append(scores)
        return ret
    
    def gpt_complete_combine_choices(self, subset, samples, config, n=3, add_angle=False, print_prompt=False):
        correct = 0
        end = 50
        for i in tqdm(subset[:end]):
            sample = samples[str(i)]
            correct += self.prompting(sample, config, n=n, add_angle=add_angle, print_prompt=print_prompt, i=str(i))
               
        print("Accuracy: ", correct/len(subset[:end]))

    def prompting(self, sample, config, n=3, add_angle=False,  print_prompt=False, i=None):
        question_prefix = "Select the correct choice for this problem:\n"
        one_shot_prefix = '''Solve the Raven Progressive Matrices problem. For example,
Q: row1:(6,0.6,50),(5,0.6,70),(4,0.6,90)
row2:(6,0.4,0),(5,0.4,20),(4,0.4,40)
row3:(6,0.5,20),(5,0.5,40),
Choices:(4,0.5,60),(4,0.5,0),(4,0.1,60),(6,0.5,60),(3,0.5,60),(5,0.5,60),(4,0.5,90),(7,0.5,60)
A:(4,0.5,60)\n'''
        cot_prompting_prefix = '''Solve the Raven Progressive Matrices problem. For example,
Q: row1:(6,0.6,50),(5,0.6,70),(4,0.6,90)
row2:(6,0.4,0),(5,0.4,20),(4,0.4,40)
row3:(6,0.5,20),(5,0.5,40),
Choices:(4,0.5,60),(4,0.5,0),(4,0.1,60),(6,0.5,60),(3,0.5,60),(5,0.5,60),(4,0.5,90),(7,0.5,60)
A: Let's solve step by step. The first number in each set decreases by 1 for each row. So, for row 3, the first number should indeed be 6-1-1=4. The second number in each set remains constant within each row. Therefore, for the third item in row 3, the second number should remain at 0.5, consistent with the rest of the row. The third number increases by 20 each time within a row. In row 3, the sequence starts at 20, then 40, and logically, the next number should be 20 + 20 + 20 = 60. Therefore, the answer is (4,0.5,60)\n'''
        two_shot_prefix = '''Another example is that,
Q: row1:(4,0.4,50),(3,0.5,30),(6,0.6,10)
row2:(6,0.4,40),(4,0.5,20),(3,0.6,0)
row3:(3,0.1,40),(6,0.2,20),
Choices:(4,0.3,0),(4,0.3,10),(4,0.3,80),(4,0.2,0),(6,0.3,0),(4,0.3,70),(4,0.3,40),(3,0.3,0)
A:(4,0.3,0)\n'''
        rpm = RPM(sample, config, n=n, add_angle=add_angle)
        questions = rpm.context.replace(' ','').split(";")
        questions = [line.strip() for line in questions]
        questions = "\n".join(questions)
        self.context = self.prefix + rpm.context
        prompt = one_shot_prefix + question_prefix + questions + '\nChoices:' + ','.join(rpm.choices)
        if i != None:
            print(f"Q{i}")
        print(prompt)
        res = self._gpt_complete(prompt)
        print(res == rpm.choices[0])
        print("----")
        return 1 if res == rpm.choices[0].strip() else 0
    
    def _gpt_complete(self, prompt):
        ret = {}
        response = openai.Completion.create(model="text-davinci-003",
                                            prompt=prompt,
                                            temperature=0,
                                            max_tokens=300,
                                            top_p=1,
                                            frequency_penalty=0,
                                            presence_penalty=0,
                                            echo=True)
        if response["choices"][0]["text"][-1] == '.':
            gpt_choice = gpt_choice = response["choices"][0]["text"][-11:-1].strip()
        else:
            gpt_choice =response["choices"][0]["text"][-10:].strip()
        print("GPT choice: ", gpt_choice)
        return gpt_choice

    def _gpt(self, prompt):
        ret = {}
        response = openai.Completion.create(model="text-davinci-002",
                                            prompt=prompt,
                                            temperature=0,
                                            max_tokens=0,
                                            top_p=1,
                                            logprobs=5,
                                            frequency_penalty=0,
                                            presence_penalty=0,
                                            echo=True)
        logprobs = response["choices"][0]["logprobs"]
        i = logprobs["text_offset"].index(len(self.context)-1)
        for k in ["tokens", "token_logprobs"]:
            ret[k] = logprobs[k][i:]
        return ret

    def _opt(self, prompt):
        ret = {}
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0][1:])
        input_ids = input_ids.to("cuda")
        token_logprobs = []
        logits = self.model(input_ids).logits
        all_tokens_logprobs = log_softmax(logits.double(), dim=2)
        for k in range(1, input_ids.shape[1]):
            token_logprobs.append(all_tokens_logprobs[:,k-1,input_ids[0,k]])
        token_logprobs = [lp.detach().cpu().numpy()[0] for lp in token_logprobs]
        i = len(self.tokenizer(self.context, return_tensors="pt").input_ids[0]) - 2
        return {"tokens": tokens[i:], "token_logprobs": token_logprobs[i:]}
    
    def _mistral(self, prompt):
        ret = {}
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0][1:])
        token_logprobs = []
        logits = self.model(input_ids).logits
        all_tokens_logprobs = log_softmax(logits.double(), dim=2)
        for k in range(1, input_ids.shape[1]):
            token_logprobs.append(all_tokens_logprobs[:,k-1,input_ids[0,k]])
        token_logprobs = [lp.detach().numpy()[0] for lp in token_logprobs]
        i = len(self.tokenizer(self.context, return_tensors="pt").input_ids[0]) - 2
        return {"tokens": tokens[i:], "token_logprobs": token_logprobs[i:]}
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name")
    parser.add_argument("--api_key")
    parser.add_argument("--config")
    parser.add_argument("-b", type=int)
    parser.add_argument("-n", type=int)
    parser.add_argument("-k", type=int, default=0)
    parser.add_argument("--prompt_dir")
    parser.add_argument("--load_dir")
    parser.add_argument("--save_dir")
    parser.add_argument("--add_angle", action='store_true')
    parser.add_argument("--subset_dir", default="subset.json")
    parser.add_argument("--experiment_name")
    parser.add_argument("--print_prompt", action='store_true')
    parser.add_argument("--is_gpt_combine_choices", type=bool, default=False)

    args = parser.parse_args()
    model, tokenizer = None, None
    if args.model_name == "gpt-3":
        openai.api_key = args.api_key
    if args.model_name[:3] == "opt":
        # free_in_GB = int(torch.mps.mem_get_info()[0]/1024**3)
        # max_memory = f'{free_in_GB-2}GB'
        # n_gpus = torch.mps.device_count()
        model = AutoModelForCausalLM.from_pretrained("facebook/"+args.model_name,
                                                     device_map='auto', offload_folder="offload", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("facebook/"+args.model_name,
                                                  use_fast=False)
    if args.model_name == "mistral":
        torch.cuda.empty_cache()
        free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
        max_memory = f'{free_in_GB-2}GB'
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}
        print(max_memory)
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
                                                     device_map='auto',
                                                     load_in_8bit=True,
                                                     max_memory=max_memory)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",
                                                  use_fast=False)
                                                  
        
    if args.model_name == "mistral_instruct":
        torch.cuda.empty_cache()
        free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
        max_memory = f'{free_in_GB-2}GB'
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}
        print(max_memory)
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",
                                                     device_map='auto',
                                                     load_in_8bit=True,
                                                     max_memory=max_memory)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",
                                                  use_fast=False)
    
    if args.prompt_dir:
        try:
            with open(args.prompt_dir, 'r') as file:
                examples = file.read().strip().split('\n')
                prefix = '\n'.join(examples[:args.k]) if args.k > 0 else ''
        except Exception as e:
            print(f"Error reading from prompt_dir {args.prompt_dir}: {e}")
            prefix = "let's think step by step. "  # default prefix
    else:
        prefix = "let's think step by step. "  # default prefix
    prefix = prefix + "let's think step by step if the following matches the established patterns.\n New Example -> "
    s = Solver(args.model_name, model=model, tokenizer=tokenizer, prefix=prefix)
    s(args.config, args.load_dir, args.save_dir, b=args.b, n=args.n, add_angle=args.add_angle, subset_dir=args.subset_dir, experiment_name=args.experiment_name, print_prompt=args.print_prompt, gpt_combine_choices=args.is_gpt_combine_choices)
    return


if __name__ == "__main__":
    main()