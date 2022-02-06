import torch

import data_util
import lightning_wrapper
import model


if True:
    sequence_length = 64
    embed_dim = 192
    indexer = data_util.SymbolIndexer()
    
    function = model.BasicModel(sequence_length, indexer, embed_dim)
    function = lightning_wrapper.LightningWrapper.load_from_checkpoint(input("path: "), map_location="cpu", f=function)

    prompt = open("prompt.txt").read()

    if len(prompt) >= sequence_length:
      prompt = prompt[-sequence_length:]
    padded_prompt = prompt.ljust(sequence_length, '*')


    x = torch.ByteTensor([indexer.to_index(symbol) for symbol in padded_prompt]).unsqueeze(0)
    print (x)


    y_pred = function(x).squeeze(0)
    y_pred = y_pred[-1]

    print(function.f.embed.interpret(y_pred, k=10))
