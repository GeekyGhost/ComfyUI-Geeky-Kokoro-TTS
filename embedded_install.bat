@echo off
echo Installing WhisperSpeech for ComfyUI...
cd /d "%~dp0"
..\..\..\python_embeded\python.exe -m pip install WhisperSpeech
..\..\..\python_embeded\python.exe -m pip install nltk
echo Installation complete!
pause