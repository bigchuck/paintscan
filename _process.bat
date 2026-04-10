setlocal
set BASE=c:\data\paintscan\data\work
set GIMP="c:\Users\charl\AppData\Local\Programs\GIMP 2\bin\gimp-console-2.10.exe"
set KEEPER="%BASE%\keepers"
set NAME=%1
set INIMAGE=%BASE%\%NAME%.jpg
set OUTDIR=%BASE%\temp
set MASTERIMAGE=%OUTDIR%\%NAME%_master.jpg
set MASTER600IMAGE=%OUTDIR%\%NAME%_master_600.jpg

set "EDGEIMAGE=%OUTDIR%\%NAME%_master_debug_edges.jpg"
set "INVERTEDIMAGE=%OUTDIR%\%NAME%_master_inverted.jpg"
set "EDGEIMAGE_GIMP=%OUTDIR:\=/%/%NAME%_master_debug_edges.jpg"
set "INVERTEDIMAGE_GIMP=%OUTDIR:\=/%/%NAME%_master_inverted.jpg"

call .venv\Scripts\activate

python main.py %INIMAGE% --out %OUTDIR%
python main.py %MASTERIMAGE% --out %OUTDIR% --debug

%GIMP% -i -b "(let* ((img (car (gimp-file-load RUN-NONINTERACTIVE \"%EDGEIMAGE_GIMP%\" \"%EDGEIMAGE_GIMP%\"))) (drawable (car (gimp-image-get-active-layer img)))) (gimp-invert drawable) (gimp-file-save RUN-NONINTERACTIVE img drawable \"%INVERTEDIMAGE_GIMP%\" \"%INVERTEDIMAGE_GIMP%\") (gimp-image-delete img)) (gimp-quit 0)"

if not exist "%KEEPER%" mkdir "%KEEPER%"

if not exist "%INVERTEDIMAGE%" (
    echo ERROR: Final file not found: %INVERTEDIMAGE%
    exit /b 1
)

copy /y "%INVERTEDIMAGE%" "%KEEPER%"
if errorlevel 1 (
    echo ERROR: Copy failed.
    exit /b 1
)

copy /y "%MASTER600IMAGE%" "%KEEPER%"
if errorlevel 1 (
    echo ERROR: Copy failed.
    exit /b 1
)

endlocal