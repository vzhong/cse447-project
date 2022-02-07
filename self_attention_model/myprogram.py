#!/usr/bin/env python
import os
import string
from matplotlib import interactive
import torch
import random
from dataclasses import dataclass
from typing import List
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


import data_util
import model
import lightning_wrapper
import model


@dataclass
class MyModelConfig:
    chkpt_path: str
    dummy_prompt: str
    device: str
    sequence_length: int
    embed_dim: int

    def __post__init__(self):
        assert len(self.dummy_prompt) >= self.sequence_length


class MyModel:
    def __init__(self, config: MyModelConfig):
        self.indexer = data_util.SymbolIndexer()
        self.config = config
        temp = model.BasicModel(config.sequence_length,
                                self.indexer, config.embed_dim)
        temp = lightning_wrapper.LightningWrapper.load_from_checkpoint(
            config.chkpt_path, map_location=config.device, f=temp)
        self.my_model = temp.f

    @classmethod
    def load_test_data(cls, fname):
        with open(fname) as f:
            data = [line[:-1].lower() for line in f]
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, "wt") as f:
            for p in preds:
                f.write(f"{p}\n")

    def prediction_from_line(self, line: str, k: int) -> str:
        if len(line) >= self.config.sequence_length:
            line = line[-self.config.sequence_length:]
        else:
            line = self.config.dummy_prompt[-(
                self.config.sequence_length - len(line)):] + line
        x = torch.ByteTensor([self.indexer.to_index(symbol)
                              for symbol in line]).unsqueeze(0)
        y_pred = self.my_model(x).squeeze(0)[-1]
        result = self.my_model.embed.interpret(y_pred, k=k+1)
        result = [c for c in result if c is not None][:k]
        return "" .join(result)

    def run_pred(self, data: List[str]):
        # your code here
        preds: List[str] = []
        for line in data:
            preds.append(self.prediction_from_line(line, k=3))
        return preds


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=(
        "train", "test", "interactive"), help="what to run")
    parser.add_argument("--work_dir", help="where to save", default="work")
    parser.add_argument("--test_data", help="path to test data",
                        default="example/input.txt")
    parser.add_argument(
        "--test_output", help="path to write test predictions", default="pred.txt")
    args = parser.parse_args()

    CONFIG = MyModelConfig(
        chkpt_path="epoch.1-step.573439.ckpt",
        dummy_prompt="in other words, living an eternity of just about anything is now more terrifying to me than death. ",
        device="cpu",
        sequence_length=64,
        embed_dim=192,
    )

    if args.mode == "train":
        if not os.path.isdir(args.work_dir):
            print("Making working directory {}".format(args.work_dir))
            os.makedirs(args.work_dir)
    elif args.mode == "test":
        model = MyModel(CONFIG)
        print("Loading test data from {}".format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print("Making predictions")
        pred = model.run_pred(test_data)
        print("Writing predictions to {}".format(args.test_output))
        assert len(pred) == len(test_data), "Expected {} predictions but got {}".format(
            len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    elif args.mode == "interactive":
        model = MyModel(CONFIG)
        user_prompt = ""
        while True:
            print(model.prediction_from_line(user_prompt, k=3))
            user_prompt += input(user_prompt)

    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))
