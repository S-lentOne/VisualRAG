#!/usr/bin/env bash

set -e

source bin/bin/activate

cd frontend

npm run tauri dev
