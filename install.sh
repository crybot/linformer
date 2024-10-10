#!/bin/bash

while read requirement; do
   pip install --no-deps --no-compile --prefer-binary --no-input --require-virtualenv "$requirement" || true
done < requirements.txt
