# F58 — Entrada de una grabación CSV, sin ZIP

Fecha: 2026-10-28

Base: rama F57 del lanzador de análisis local.

## Cambio aplicado

- El ejecutor vigente acepta una grabación CSV sin comprimir por ejecución, para una serie formal R1–R5. La GUI solo ofrece archivos `.csv`; la opción CLI `--zip` ya no existe.
- Los ZIP se rechazan antes del procesamiento. También se detecta un ZIP renombrado con extensión `.csv`; no se inspeccionan sus miembros, no se extraen y no se crean carpetas de salida.
- Se renombró el módulo a `src/analizar_csv_emg.py` para que su nombre coincida con la entrada admitida. `INICIAR_INFERENCIA.bat` inicia ese módulo y deja clara la restricción.
- README y los guiones de presentación describen la misma conducta. `reports/F57_ONECLICK_STATUS.md` se conserva como registro histórico del estado previo.

## Alcance que permanece

La ejecución sigue siendo descriptiva: produce características y eventos candidatos de actividad. No carga un modelo; no cuenta repeticiones mecánicas verificadas ni predice fatiga fisiológica. Para procesar más de una grabación, se ejecuta el flujo por separado con cada CSV autorizado.

## Verificación

- Pruebas focales de entrada CSV, rechazo de ZIP y flujo CSV de extremo a extremo: **8/8** aprobadas.
- Suite completa: **100/100** aprobadas; `compileall` terminó correctamente.
- `git diff --check` terminó con código 0; mostró solo avisos de normalización LF/CRLF en los archivos Windows.
- Manifiesto público: **74/74** hashes SHA-256 coinciden con los blobs Git canónicos.
- La presentación Canva existente conserva sus 10 diapositivas; la conclusión final ahora indica un CSV individual R1–R5 por ejecución y el rechazo de ZIP.
- La prueba incluye ZIP con extensión `.zip`, ZIP renombrado como `.csv`, y la opción CLI `--zip`; en ninguno se genera salida.
- No se procesaron señales reales durante esta corrección; el extremo a extremo ejercita un CSV sintético de software.

## Límites

No se ha ejecutado una inferencia sobre CSV de los dos pacientes: esos archivos individuales no estaban disponibles. El ZIP no es compatible con esta herramienta. Esta corrección no convierte la ruta descriptiva en inferencia de modelo ni en validación externa.
