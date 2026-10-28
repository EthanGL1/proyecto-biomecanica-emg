@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY_CMD="
where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 --version >nul 2>nul
  if not errorlevel 1 set "PY_CMD=py -3.12"
)

if not defined PY_CMD if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY_CMD="%LOCALAPPDATA%\Programs\Python\Python312\python.exe""
if not defined PY_CMD if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY_CMD="%LOCALAPPDATA%\Programs\Python\Python313\python.exe""
if not defined PY_CMD if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" set "PY_CMD="%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe""

if not defined PY_CMD (
  echo No encontre Python 3.12 o posterior.
  echo Instala Python 3.12+ desde https://www.python.org/downloads/windows/ y vuelve a ejecutar este archivo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creando el entorno local con %PY_CMD% ...
  %PY_CMD% -m venv .venv
  if errorlevel 1 goto failed
)

echo Instalando las dependencias fijadas en requirements.txt ...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo Comprobando dependencias y selector de archivos ...
.venv\Scripts\python.exe -c "import numpy,pandas,PIL,tkinter; print('NumPy',numpy.__version__,'| pandas',pandas.__version__,'| Pillow',PIL.__version__,'| Tkinter disponible')"
if errorlevel 1 goto failed

echo Verificando sintaxis del codigo ...
.venv\Scripts\python.exe -m compileall -q src
if errorlevel 1 goto failed

echo Ejecutando pruebas automatizadas ...
.venv\Scripts\python.exe -m unittest discover -s tests -q
if errorlevel 1 goto failed

echo.
echo Instalacion y comprobaciones completadas.
pause
exit /b 0

:failed
echo.
echo La instalacion o una comprobacion fallo. Conserva el mensaje anterior para diagnosticarlo.
pause
exit /b 1
