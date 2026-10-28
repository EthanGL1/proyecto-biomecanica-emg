@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo El entorno local no existe. Voy a instalarlo y verificar las dependencias primero.
  call INSTALAR_TODO.bat
  if errorlevel 1 exit /b 1
)

echo Este flujo procesa descriptivamente CSV EMG o ZIPs que contienen CSVs.
echo No carga un modelo, no cuenta repeticiones verificadas ni predice fatiga fisiologica.
echo Seleccione un CSV formal R1-R5 o un ZIP con grabaciones CSV compatibles.
echo.
.venv\Scripts\python.exe src\analizar_zip_emg.py --pick
if errorlevel 1 (
  echo.
  echo El proceso no termino correctamente. Revise la entrada y conserve el mensaje de error.
  pause
  exit /b 1
)
pause
exit /b 0
