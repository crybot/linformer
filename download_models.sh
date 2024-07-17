#!/bin/bash

declare -a FILES_LM=(
"model.safetensors"
"config.json"
"generation_config.json"
"merges.txt"
"vocab.json"
"tokenizer.json"
"tokenizer_config.json"
"generation_config_for_summarization.json"
)

declare -a MODELS=( "openai-community/gpt2" "facebook/bart-base" "facebook/bart-large-cnn" )

function validate_url(){
  if [[ `wget -S --spider $1  2>&1 | grep 'HTTP/1.1 200 OK'` ]]; then echo "true"; fi
}

PREFIX="https://huggingface.co"
for model in "${MODELS[@]}"; do
  for file in "${FILES_LM[@]}"; do

    DOWNLOAD_DIR="./models/$model"
    mkdir -p "$DOWNLOAD_DIR"

    # Check if the file already exists
    if [ -f "$DOWNLOAD_DIR/$file" ]; then
      echo "File already exists: $DOWNLOAD_DIR/$file. Skipping download."
      continue
    fi

    # Check if the file exists on HuggingFace, if not then just skip it for this
    # model (most likely it's not needed)
    URL="${PREFIX}/${model}/resolve/main/${file}"
    if [ ! `validate_url "$URL"` ]; then
      echo "File not hosted on HuggingFace, skipping."
      continue
    fi

    # Download on a temporary file so that interrupted downloads can be resumed later
    TMPFILE="${DOWNLOAD_DIR}/${file}.part"
    wget --continue -O "$TMPFILE" "$URL"

    # Move completed download to the right location
    if [ -f "$TMPFILE" ]; then
      mv "$TMPFILE" "${DOWNLOAD_DIR}/${file}"
    fi

  done
done
