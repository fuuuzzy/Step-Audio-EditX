#!/bin/bash

uv sync

git lfs install

git clone https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer  models/Step-Audio-Tokenizer
git clone https://huggingface.co/stepfun-ai/Step-Audio-EditX models/Step-Audio-EditX