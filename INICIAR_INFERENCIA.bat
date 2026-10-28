@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo El entorno local no existe. Voy a instalarlo y verificar las dependencias primero.
  call INSTALAR_TODO.bat
  if errorlevel 1 exit /b 1
)

echo Este analisis calcula caracteristicas descriptivas y eventos candidatos de actividad.
echo No cuenta repeticiones mecanicas ni predice fatiga fisiologica.
echo Seleccione un CSV sin comprimir de una grabacion formal R1-R5.
echo No se admiten ZIP; procese cada CSV individual por separado.
echo.
.venv\Scripts\python.exe src\analizar_csv_emg.py --pick
if errorlevel 1 (
  echo.
  echo El proceso no termino correctamente. Revise la entrada y conserve el mensaje de error.
  pause
  exit /b 1
)
pause
exit /b 0
