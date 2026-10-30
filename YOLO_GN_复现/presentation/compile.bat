@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .build mkdir .build
"D:\texlive\2024\bin\windows\xelatex.exe" -interaction=nonstopmode -halt-on-error -output-directory=.build -jobname=YOLO_GN main.tex || exit /b 1
"D:\texlive\2024\bin\windows\xelatex.exe" -interaction=nonstopmode -halt-on-error -output-directory=.build -jobname=YOLO_GN main.tex || exit /b 1
"D:\texlive\2024\bin\windows\xelatex.exe" -interaction=nonstopmode -halt-on-error -output-directory=.build -jobname=YOLO_GN main.tex || exit /b 1
python pdf_to_image_pptx.py || exit /b 1
echo PDF and image-only PPTX created in the project folder.

