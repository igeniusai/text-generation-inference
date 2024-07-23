#!/bin/bash

ldconfig 2>/dev/null || echo 'unable to refresh ld cache, not a big deal in most cases'

CARGS="${CUSTOM_ARGS:=}"

if [[ -z "${HF_MODEL_ID}" ]]; then
  text-generation-launcher $CARGS $@
else
  text-generation-launcher $CARGS --model-id "${HF_MODEL_ID}" $@
fi