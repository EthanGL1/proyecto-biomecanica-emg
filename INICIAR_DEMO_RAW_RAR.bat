@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "RAW_ARCHIVE=%~1"
if not defined RAW_ARCHIVE set "RAW_ARCHIVE=%USERPROFILE%\Downloads\raw.rar"

if not exist "%RAW_ARCHIVE%" (
  echo No encuentro el RAR: "%RAW_ARCHIVE%"
  echo Uso: INICIAR_DEMO_RAW_RAR.bat "C:\ruta\raw.rar"
  pause
  exit /b 2
)

if not exist ".venv\Scripts\python.exe" (
  echo El entorno local no existe. Voy a instalarlo y verificar las dependencias primero.
  call INSTALAR_TODO.bat
  if errorlevel 1 exit /b 1
)

echo El RAR original se conserva sin cambios.
echo Se abrirá el índice HTML local con el análisis descriptivo.
echo No se carga un modelo de IA ni se predicen repeticiones o fatiga.
echo.
.venv\Scripts\python.exe src\ejecutar_demo_raw_rar.py --archive "%RAW_ARCHIVE%"
if errorlevel 1 (
  echo.
  echo El proceso no terminó. Conserva el mensaje de error y revisa la estructura RAR/CSV.
  pause
  exit /b 1
)
pause
exit /b 0
