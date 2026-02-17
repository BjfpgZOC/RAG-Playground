#!/bin/sh
. ./set_vars.sh
uvicorn rag.api.main:app --host 0.0.0.0 --port 8000 --reload