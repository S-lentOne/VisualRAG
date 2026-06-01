@echo off

call bin\Scripts\activate.bat

cd frontend

npm run tauri dev
