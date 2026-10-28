# F59 — CSV y ZIP en el lanzador descriptivo

**Fecha:** 2026-10-28
**Estado:** implementado en la rama `f57-oneclick-zip-analysis`; propuesta de actualización pendiente de revisión/merge mediante PR #3.

## Cambio realizado

El lanzador existente ahora comparte un único procesador para un CSV formal o un ZIP con varios CSV. El adaptador `src/analizar_csv_emg.py` conserva compatibilidad con comandos previos y reexporta la implementación común de `src/analizar_zip_emg.py`. `INICIAR_INFERENCIA.bat` ofrece selección por archivo y mantiene una salida privada local. Se conserva la presentación existente y se actualiza el guion de demostración a CSV/ZIP.

Para ZIP, el lector valida las rutas antes de procesar, rechaza rutas inseguras, symlinks, miembros cifrados y duplicados normalizados, impone límites publicados y copia cada miembro admisible a un nombre temporal generado por el programa. No extrae con rutas controladas por el archivo. El manifiesto de resultados no publica los nombres de las carpetas dentro del ZIP.

## Datos y alcance admitidos

- Un CSV con columnas `Relative_Time_s,ADC_Value[,Voltage_mV]`, o un ZIP que contenga uno o más de esos CSV.
- Se procesan series formales R1–R5 con serie inequívoca; lateralidad se identifica cuando el nombre o carpeta lo permite.
- R0/R0_1 quedan como registros preliminares de QC y no entran en resultados formales. Nombres ambiguos, serie ausente y esquemas no soportados quedan sin procesar y se registran.
- Cada corrida escribe resumen, manifiesto y resultados descriptivos por grabación en el destino local elegido.

## Qué no hace

No hay un checkpoint de modelo de fatiga/repeticiones disponible en el paquete público ni cargado por este lanzador. Los eventos son candidatos de actividad; no son repeticiones mecánicas verificadas. El programa no emite puntuaciones, niveles, probabilidades, localización temporal o diagnóstico de fatiga. El benchmark temporal existente clasifica tercios de una grabación y no valida fatiga. No se procesó ningún dato identificable ni se hizo inferencia sobre el ZIP de dos pacientes.

`Voltage_mV` es un campo del archivo, pero no se asume como calibración física. Las conclusiones dependen de la señal guardada y no reemplazan una validación fisiológica independiente.

## Verificación ejecutada

- `python -m unittest discover -s tests -v`: **112 pruebas aprobadas** (incluye casos de CSV/ZIP, fin-a-fin sintético, rechazo de ZIP renombrado, seguridad de rutas, R0/R0_1, esquema, pseudonimización y comprobación de `model_loaded=false`).
- `python presentation/generate_10min_script_pdf.py`: generó el PDF del guion inglés F59.
- `git diff --check`: sin errores de espacios.
- No se ejecutó `pytest` porque no está instalado en el entorno virtual; se utilizó el descubridor `unittest` incluido en Python, que ejecutó la suite completa.
- Las pruebas con entradas EMG son sintéticas y verifican el software; no constituyen validación fisiológica, prueba en pacientes nuevos ni inferencia de modelo.

## Instrucciones de uso

1. Doble clic en `INSTALAR_TODO.bat` la primera vez.
2. Doble clic en `INICIAR_INFERENCIA.bat`.
3. Seleccionar un CSV compatible o un ZIP con CSV compatibles.
4. Seleccionar una carpeta local privada para resultados; conservar la salida en almacenamiento autorizado.

Desde PowerShell abierto en la carpeta del proyecto:

```powershell
.\.venv\Scripts\python.exe src\analizar_zip_emg.py --csv "C:\ruta\privada\iz_R1.csv" --output-dir "$env:LOCALAPPDATA\EMG_Biomecanica\Resultados"
.\.venv\Scripts\python.exe src\analizar_zip_emg.py --zip "C:\ruta\privada\nuevos_participantes.zip" --output-dir "$env:LOCALAPPDATA\EMG_Biomecanica\Resultados"
```

`--output-dir` puede omitirse para usar el directorio local predeterminado. El modo de entrada CSV y ZIP es mutuamente excluyente. No incluyas `participants.csv` ni identificadores personales en las columnas del CSV.

## Archivos de esta actualización

- `src/analizar_zip_emg.py`: procesador compartido y lector seguro CSV/ZIP.
- `src/analizar_csv_emg.py`: adaptador de compatibilidad de CLI.
- `tests/test_analizar_csv_emg.py` y `tests/test_analizar_zip_emg.py`: pruebas de integración y seguridad.
- `INICIAR_INFERENCIA.bat`, `README.md` y guiones F59 en `presentation/`.
- `presentation/Canva_10_minute_speaker_script_EN_F59.pdf`: PDF reproducible del guion hablado.

No se incluyeron señales locales, identificadores, etiquetas individuales, predicciones individuales ni pesos de modelos.
