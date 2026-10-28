# F57 — Lanzador de análisis EMG por ZIP y CSV individual

> **Registro histórico:** esta nota describe la implementación original F57. La entrada ZIP fue retirada en la corrección posterior; consulta `reports/F58_CSV_ONLY_UPDATE.md` para el comportamiento vigente.

Fecha: 2026-10-28
Base de código: F56 (`71f0ed3de751cf69cb47ba22de12d2634a9481d3`)
Rama local: `f57-oneclick-zip-analysis`

## Qué quedó implementado

- `src/analizar_zip_emg.py` inspecciona ZIPs sin extraer rutas suministradas por el archivo, valida rutas y esquema CSV, rechaza miembros ambiguos/inseguros, conserva R0/R0_1 como controles preliminares y dirige cada CSV formal compatible al procesador existente `src/reproducir_emg_referencia.py`. También acepta un CSV R1–R5 individual, sin inferir identidad ni agrupamiento de participante.
- El flujo produce un manifiesto por grabación, una tabla agregada de eventos candidatos, resúmenes y tableros descriptivos individuales. Los identificadores de salida son REC y, cuando la estructura lo permite, P01…; no se copian los nombres de los miembros del ZIP a las tablas.
- `INSTALAR_TODO.bat` crea `.venv`, instala las versiones declaradas de las dependencias directas y ejecuta compilación y pruebas. `INICIAR_INFERENCIA.bat` abre el selector de ZIP o CSV/carpeta y lanza el flujo; el nombre no implica que haya un modelo de inferencia disponible. También admite la CLI `--csv RUTA` o `--zip RUTA`.
- El selector de salida abre en `%LOCALAPPDATA%\EMG_Biomecanica\Resultados` cuando esa variable de Windows está disponible, con alternativa local en el perfil del usuario. El operador aún debe evitar destinos compartidos/sincronizados: no se inspeccionan ACLs ni políticas del host.
- La configuración de VS Code apunta al entorno `.venv`. El README mantiene intacta su sección previa `Límites y uso`.

## Verificación ejecutada

- La instalación local del entorno Python 3.12.14 terminó; se instalaron NumPy 2.3.5, pandas 3.0.1 y Pillow 12.3.0; Tkinter importó correctamente.
- `python -m compileall -q src tests`: correcto después de conectar la entrada CSV individual.
- `python -m unittest discover -s tests -q`: 104 pruebas aprobadas en el entorno Python 3.12.14, incluyendo el flujo completo desde `--csv`.
- Prueba de integración CSV individual: se generó una señal sintética de software de 5 s a 1 kHz en un directorio temporal y se ejecutó la CLI `--csv`; creó `run_manifest.csv`, `run_summary.json` y `recordings/REC0001/dashboard.html`, confirmó `model_loaded=false` y devolvió `null` para predicciones mecánicas y fisiológicas. Es una prueba funcional de software, no evidencia fisiológica.
- Prueba real de extremo a extremo, con un CSV R1 seleccionado del ZIP de desarrollo y empaquetado temporalmente bajo un pseudónimo: 1/1 registro procesado; generó el resumen, el manifiesto, el índice HTML y el tablero. El conteo de eventos candidatos permanece en el recibo local restringido. Ese recibo informa `model_loaded=false`, sin conteo de repeticiones mecánicas y sin predicción de fatiga. No se conserva ni publica el CSV derivado ni el conteo individual de prueba.
- La revisión dirigida de cambios de Codex Security (scan `51bb3bda-b9fa-4664-a6c3-a1112b81f4b4`) se completó: 0 hallazgos reportables en el diff y cobertura completa de las superficies registradas para este cambio (procesador CSV, lanzador y materiales de uso/pruebas). No es una prueba de penetración ni una auditoría completa del repositorio. No evaluó ACL del equipo, volúmenes temporales, sincronización, índice de paquetes o dependencias transitivas; el ZIP independiente de dos pacientes tampoco estaba disponible.
- La comparación directa con la base de esta rama confirma que el bloque existente `## Límites y uso` permanece sin cambios.

## Estado científico y de validación

El inventario previo del proyecto contiene 138 grabaciones formales R1–R5 y dos registros R0/R0_1 de control, en 15 grupos derivados de carpetas. Los archivos disponibles no incluyen marcas sincronizadas de repeticiones ni etiquetas independientes de fatiga del cuádriceps. Por ello no hay base para seleccionar un modelo por sensibilidad/F1 de repeticiones o fatiga, ni para presentar niveles por repetición como validados. Los eventos del programa son candidatos de actividad basados en el pipeline descriptivo, no repeticiones verificadas.

El ZIP independiente de dos pacientes no estaba disponible en las ubicaciones inspeccionadas. El único ZIP de Descargas inspeccionado no contenía CSV. La prueba real descrita arriba usó una grabación de desarrollo y no sustituye una validación externa. Se requiere que el usuario proporcione el ZIP de dos pacientes con autorización de uso para ejecutar esa prueba; no se usará para seleccionar parámetros ni reentrenar.

El checkpoint Borg esperado en el recibo anterior continúa ausente y, en cualquier caso, su tarea documentada es auto-reporte de esfuerzo de hombro, no detección de repeticiones o fatiga fisiológica de cuádriceps. Este cambio no selecciona ni declara un modelo ganador.

## Presentación y guion F57

- Se actualizó el diseño Canva existente de diez diapositivas. La revisión corrige los denominadores evaluables por métrica (R1-R5: RMS 26 pares/15 participantes; MDF/MNF 20/12; bilateral: RMS 60/14 y MDF/MNF 49/12), identifica Otsu como segmentación exploratoria de actividad y aclara que el lanzador F57 no carga un clasificador.
- La diapositiva final ahora describe la ejecución local con `INICIAR_INFERENCIA.bat` como análisis descriptivo. No se agregaron señales, filas individuales, identidades, conteos privados ni resultados de los dos pacientes ausentes.
- El guion en inglés conserva el orden de las diez diapositivas y asigna tiempos que suman 10:00. Incluye ecuaciones verbales, filtros y sus parámetros, cifras agregadas con denominadores métricos, instrucciones de demostración CSV/ZIP y la limitación del ZIP de dos pacientes no suministrado. Se añadió también `presentation/Guion_presentacion_10_minutos_ES_F57.md`, con los mismos resultados y límites para exposición oral en español; sus diez tiempos suman 10:00. El PDF del guion inglés se genera desde Markdown con `presentation/generate_10min_script_pdf.py`; `presentation/requirements-pdf.txt` fija la dependencia opcional.
- Se evitó presentar el Macro-F1 temporal (0.716 +/- 0.146) como detección de fatiga. El benchmark es previo, usa tercios temporales y no se carga desde el lanzador F57.

## Pendiente antes de afirmar la entrega integral

1. Ejecutar la inferencia descriptiva con el ZIP independiente de dos pacientes y verificar estructura/etiquetas, sin ajustar el sistema con ese conjunto.
2. Conseguir referencias mecánicas sincronizadas para validar conteos de repeticiones y etiquetas independientes para validar fatiga; de otro modo, mantener los resultados como eventos y tendencias exploratorias de señal.
3. La rama F57 se publicó como `f57-oneclick-zip-analysis`; el PR #3 está abierto y aún no se ha integrado en `main`. La revisión de seguridad citada arriba cubre el diff, no una prueba de penetración ni las políticas del equipo; conserva los datos de entrada y las salidas en ubicaciones autorizadas y restringidas.
4. La presentación y el guion ya fueron actualizados con evidencia disponible; incorporar resultados de los dos pacientes solo cuando se suministre un ZIP autorizado y compatible. No llamar a esa ejecución validación de fatiga sin etiquetas independientes.

No se incluyen datos crudos, salidas por grabación, checkpoints, participantes ni predicciones individuales en esta rama.
