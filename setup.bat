@echo off

echo Creating Python virtual environment...
python -m venv bin

call bin\Scripts\activate.bat

echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Installing frontend dependencies...
cd frontend
npm install

echo.
echo Setup complete.
pause
