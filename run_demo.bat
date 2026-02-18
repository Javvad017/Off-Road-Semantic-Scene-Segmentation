@echo off
title Off-Road Semantic AI SegFormer
color 0f
echo Starting AI Environment...
echo Using Python: %python%
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
python app.py
pause
