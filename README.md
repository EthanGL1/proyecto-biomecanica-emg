# Análisis EMG — paquete público reproducible F54 

## Actualizacion F56 - visor y estado de auditoria

F56 corrige la comprobacion de disponibilidad de DecompressionStream del visor: los navegadores compatibles ahora decodifican el payload y los que no lo soportan muestran el aviso existente antes de inicializar la interfaz. Se agrego una prueba de regresion para ambas ramas.

La verificacion local de esta revision obtuvo 92/92 pruebas, compilacion Python correcta, prueba de ejecucion JavaScript con la API presente y ausente, y git diff --check sin errores. La revision cruzada del codigo no encontro otro problema en este cambio. El escaneo formal Codex Security quedo sin resultado final porque el control de aplicaciones bloqueo el modulo nativo al registrar el inventario; no se afirma una auditoria formal sin hallazgos.

F56 no cambia datos, resultados, etiquetas, modelos ni metricas. Se mantienen los limites cientificos y de privacidad descritos en este README.


Este repositorio reúne código de procesamiento, pruebas, bibliografía y resultados agregados autorizados. El informe DOCX/PDF F31 de reports/ es histórico; no es la versión privada editable del manuscrito para revisión ni un artículo final aprobado.

El estado público F52 (reports/F52_STATUS_20261020.md) documenta la entrada directa de fragmentos CSV crudos, las pruebas, la demostración y los límites vigentes. Los cortes F51, F50.9 y anteriores se conservan como historia.

La corrección F53 reconcilia el objetivo definido por la guía docente con lo que puede confirmarse en los CSV locales. Consulta [la nota de protocolo F53](reports/F53_PROTOCOL_RECONCILIATION.md) antes de interpretar los resultados: la guía sí especifica recto femoral durante extensión unilateral de rodilla; el archivo de señales no demuestra por sí solo que cada canal se adquirió en ese sitio ni que todos los parámetros previstos se ejecutaron.

## Actualización F54 — resultados agregados y presentación

El corte F54 añade resúmenes públicos de la exportación F55 y el guion aprobado para la presentación de 10 minutos.

En este checkout limpio, compileall terminó correctamente y la suite existente pasó 91/91 pruebas. La actualización no entrena sobre datos privados ni añade resultados por pliegue.

El procedimiento local seleccionado para posición temporal obtuvo Macro-F1 0,7164 ± 0,1465, ROC-AUC macro OvR 0,9170 ± 0,0671 y PR-AUC macro 0,8704 ± 0,0976 en 15 pliegues agrupados por participante. La etiqueta corresponde a los tercios inicial, medio y final de cada grabación, no a fatiga. El resultado F31 histórico de la misma tarea informa Macro-F1 0,740; ambos cortes se conservan con su procedencia y la diferencia entre ellos no se reconcilió en este cambio.

Borg (esfuerzo percibido de hombro, RPE ≥17) obtuvo Macro-F1 0,5787 ± 0,1344, ROC-AUC 0,7872 ± 0,1147 y PR-AUC 0,7900 ± 0,0934 en 34 pliegues. Cerqueira (auto-reporte ordinal de miembro superior, 0/1/2) obtuvo 0,3682 ± 0,0905, 0,6322 ± 0,0474 y 0,4576 ± 0,0494, respectivamente, en 13 pliegues. Su comparador diagnóstico de tiempo/tarea obtuvo 0,7135 ± 0,0834 de Macro-F1 y no es el predictor EMG ni un modelo desplegable. Las tres tareas no son comparables entre sí como si midieran el mismo fenómeno y ninguna valida fatiga del cuádriceps.

Se publican 36 filas agregadas por tarea, modelo y configuración y 36 matrices de confusión agregadas: [métricas por modelo](reports/F55_METRICS_AGGREGATED_PUBLIC.csv) y [matrices de confusión](reports/F55_CONFUSION_COUNTS_AGGREGATED_PUBLIC.csv). El CSV de 625 filas por pliegue y su manifiesto con rutas privadas no se publican. R1–R5, comparación bilateral y filtros conservan los resultados agregados ya publicados, sin recálculo en F54.

La presentación Canva “Defensa sEMG cuádriceps” quedó guardada y aprobada con 10 diapositivas. Se incluye el guion hablado en inglés: 1.238 palabras y tiempos asignados que suman 10:00 (aproximadamente 124 palabras/minuto según el conteo textual). La exportación del diseño no se incorpora al repositorio.

El DOCX completo más reciente queda fuera del paquete público porque incluye identidades de autor y resultados por grupo de participante. El informe público F31 continúa siendo histórico.

## Verificación actual F52

- La suite pública pasó 89/89 pruebas; compileall, ayuda del CLI y la demo sintética con visor terminaron correctamente.
- --input-fragments procesa fragmentos del esquema Relative_Time_s,ADC_Value[,Voltage_mV] sin añadir claves de identidad. Requiere --series, ordena por timestamps originales, rechaza solapamientos y duplicados, conserva huecos sin interpolarlos y registra archivos, muestras y huecos en el resumen.
- Con un archivo R1 real de 30.946 muestras, el procesamiento completo y la reconstitución desde tres fragmentos presentados fuera de orden produjeron siete tablas idénticas byte por byte y 295 filas de ventana. La fuente permaneció sin cambios. Las señales, fragmentos, salidas y recibo detallado permanecen en almacenamiento privado.
- La demo sintética genera diez productos, 200 filas de ventana y eventos candidatos exploratorios. No usa datos del estudio y no emite puntuaciones supervisadas.
- El snapshot público es intermedio. No se afirma validación de fatiga funcional, generalización entre datasets, resultado de escaneo formal de seguridad ni aprobación editorial.

## Alcance científico vigente

El objetivo local previsto es la sEMG del recto femoral (rectus femoris), parte del cuádriceps, durante extensión unilateral de rodilla, según la guía docente. El inventario local comprende 15 grupos derivados de nombres de archivo, con 138 grabaciones formales R1–R5 y dos registros R0/R0_1 reservados para control de calidad. El análisis de las señales es descriptivo: el archivo no incluye una etiqueta independiente y sincronizada de fatiga, fuerza/torque medidos ni marcadores mecánicos de repetición. Las marcas de actividad son candidatas; no establecen nivel, inicio o pico fisiológico de fatiga. La guía define el músculo y el protocolo previsto, pero los CSV no verifican individualmente el sitio del electrodo, la carga ejecutada, la fase/cadencia, la modalidad efectiva del canal o la causa de terminación.

Borg corresponde a esfuerzo percibido de hombro y no produjo una inferencia reproducible en el runtime autorizado disponible. Cerqueira corresponde a auto-reporte ordinal de fatiga autopercibida de miembro superior; su validación por sujeto es interna a ese corpus y el diagnóstico de tiempo/tarea supera al predictor EMG. Ninguna rama permite transferir predicciones a cuádriceps.

La fuente privada LaTeX F51 compiló en el editor integrado, pero no se exportó un PDF. La extensión de 15–20 páginas y la inspección visual por página siguen sin verificarse. Codex Security tampoco produjo un informe por un bloqueo del módulo nativo; no se declara cero hallazgos.

## Privacidad

El repositorio no contiene señales privadas, identificadores, CSV individuales, predicciones granulares, HTML con señales, checkpoints restringidos ni el manuscrito completo. La documentación pública y los ejemplos reproducibles no incluyen datos de participantes.

## Estados anteriores


## Resultados consolidados F50.5 (intermedio)

La revisión F39 v3 verificó los hashes de los 140/140 archivos crudos locales, agrupados en 15 grupos seudonimizados: 138 grabaciones formales R1–R5 y dos controles de calidad. Se obtuvieron 7.601 ventanas válidas por calidad y 2.102 candidatas de actividad. Hay 27 pares R1–R5 disponibles y 63 pares izquierda–derecha. La estadística primaria resume cambios medios agrupados por participante con bootstrap de 4.000 remuestreos.

Para R5−R1, RMS cambió +8,571 % (IC95 %: −0,681 a 18,264; 26/27 pares, 15 participantes), MDF −0,535 Hz (−3,877 a 2,663; 20/27, 12) y MNF −3,306 Hz (−6,878 a −0,277; 20/27, 12). En los contrastes espectrales normalizados dentro de serie, MDF cambió +2,482 % (−3,575 a 8,243) y MNF +2,350 % (−4,478 a 8,735), ambos en 20/27 pares y 12 participantes; ambos intervalos incluyen cero.

En el contraste bilateral izquierda−derecha, RMS fue −0,342 % (IC95 %: −9,865 a 8,837; 60/63 pares, 14 participantes), MDF +0,994 Hz (−3,969 a 5,618; 49/63, 12) y MNF +1,104 Hz (−3,751 a 5,675; 49/63, 12). Los tres intervalos incluyen cero.

La sensibilidad exploratoria comparó ventanas de 0,25, 0,5 y 1,0 s con ocupación activa mínima de 25 %, 50 % y 75 %. Con ocupación de 50 %, las medianas de cobertura para esas duraciones fueron 27,594 %, 19,109 % y 2,716 %. La retención del patrón espectral débil fue 19,23 %, 7,69 % y 0 %; la del patrón espectral tardío, 36,36 %, 19,05 % y 0 %; los pares R1–R5 evaluables fueron 19, 13 y 3. A 1,0 s la variabilidad no fue estimable por la escasez de ventanas. La ocupación activa de 50 % es distinta del solapamiento temporal de ventanas de 50 % usado en otros análisis; no son la misma condición.

Estas tendencias describen la señal almacenada. No hay etiquetas independientes de fatiga, pérdida de fuerza medida, fuerza/torque sincronizados ni marcadores mecánicos que verifiquen repeticiones; la modalidad de la señal y la calibración física del canal siguen sin confirmarse. No se concluye fatiga fisiológica ni pérdida de fuerza.

Cerqueira conserva como objetivo el auto-reporte ordinal de fatiga de miembro superior (0/1/2), separado del análisis de cuádriceps. El LOSO anidado interno comprende 13 participantes, con 146 ensayos/44.886 ventanas a cobertura ≥90 % como análisis primario y 143 ensayos/44.428 ventanas a ≥95 % como sensibilidad. Según el recibo primario F50.7 (bootstrap por participante, 5.000 remuestreos, semilla 4021), el macro-F1 medio por participante fue 0,368 (IC95 %: 0,319–0,415) para EMG y 0,713 (0,666–0,755) para el diagnóstico separado de tiempo/tarea. Las etiquetas están fuertemente asociadas a la progresión temporal; el comparador temporal es diagnóstico y no un predictor EMG desplegable.

Un replay independiente en proceso limpio reconstruyó la inferencia de un pliegue retenido desde señal cruda: 281.814 muestras, 223 ventanas completas de 1 s, aproximadamente 1.259,26 Hz y cuatro canales. Las 222 ventanas estables comparadas con las probabilidades OOF guardadas difirieron como máximo 6,65×10⁻¹²; la predicción de la señal completa coincidió exactamente con la obtenida tras reensamblar tres fragmentos cronológicos contiguos. El recibo privado se identifica por SHA-256 a5643640397d9715d27bcde5810b515b73476eb9ec2297cb66f7a4d42f990562. Es un replay interno del mismo corpus para auto-reporte ordinal de miembro superior, no validación externa ni detector de fatiga de cuádriceps. Las señales y predicciones granulares permanecen restringidas.

El checkpoint Borg sigue sin producir una inferencia reproducible: Windows Application Control bloqueó la extensión nativa requerida de SciPy antes de predecir. No hay validación fisiológica externa compatible con el objetivo local de cuádriceps. La prueba técnica privada Cerqueira full-fit con señal sintética no estima desempeño y no cambia esa conclusión. La fuente LaTeX F50.9 compiló en el editor integrado, pero no se exportó PDF; Device Guard bloqueó el ejecutable Tectonic (error 4551), de modo que la paginación y QA visual siguen pendientes. El escaneo formal Codex Security se interrumpió cuando Windows Application Control bloqueó su módulo nativo; no hay informe de hallazgos. La publicación permanece intermedia.

## Alcance y datos

El análisis local usó únicamente las grabaciones EMG: 138 registros formales R1–R5 y dos registros R0/R0_1 reservados para control de calidad, agrupados en 15 códigos derivados de los nombres de archivo. `participants.csv` y sus campos demográficos no se usaron como entradas, etiquetas ni covariables del análisis o entrenamiento. Los códigos de grabación no se contrastan aquí con una clave independiente de identidad. Las particiones del benchmark temporal mantuvieron juntas las grabaciones de cada grupo derivado.

No se publican señales, datos por participante, características, etiquetas, particiones, predicciones, resultados por pliegue ni recibos privados. Las mediciones originales y salidas con granularidad individual permanecen en el almacenamiento local restringido. El benchmark completo requiere acceso autorizado a los datos. La comprobación exploratoria externa posterior al congelamiento del modelo no se incluye como datos o figura en esta edición pública.

## Resultados históricos agregados locales (F31; clasificación de posición temporal)

Los valores siguientes pertenecen a la publicación histórica F31. El análisis F49 posterior añade el filtro de actividad, el análisis de duraciones/ocupaciones y las comparaciones emparejadas R1–R5 descritas arriba; los resultados F31 no deben interpretarse como el cierre fisiológico actual.

La selección anidada LOSO comparó 325 configuraciones de 11 familias con 15 pliegues externos por participante y cuatro pliegues internos agrupados. Para los tres tercios temporales, el procedimiento anidado obtuvo macro-F1 medio por participante 0,740 (IC 95 % bootstrap 0,689–0,787) y ROC-AUC macro OvR 0,941 (0,921–0,959). La regresión logística tuvo el mayor punto de macro-F1 entre selectores de familia fija (0,751), pero no se estableció superioridad ni un único modelo universal.

La estratificación post hoc separó las piernas registradas. La izquierda incluyó 14 grupos, 70 grabaciones y 210 segmentos; la derecha, 15 grupos, 68 grabaciones y 204 segmentos. Catorce grupos fueron pareados. Macro-F1 fue 0,776 (IC 95 % 0,675–0,874) a izquierda y 0,686 (0,611–0,762) a derecha; la diferencia pareada izquierda menos derecha fue +0,092 (−0,052 a +0,238), que incluye cero. La cobertura por serie se resume en `reports/LATERALITY_F31_20261018.md` y el heatmap agregado. Estas métricas describen clasificación de posición temporal, no fatiga ni asimetría fisiológica.

En ventanas de 1 s con 50 % de solapamiento, la rama HP20 dio pendientes medianas por grupo de +19,30 cuentas ADC/min para RMS (IC95 % +12,06 a +41,41), −52,49 Hz/min para MDF (−68,38 a −8,54) y −44,03 Hz/min para MNF (−86,01 a −22,17). Las pendientes espectrales mantuvieron dirección descendente al repetir el cálculo sin solapamiento y en las otras ramas. La linealidad fue baja (R² mediano: RMS 0,031; MDF 0,118; MNF 0,083) y las trayectorias no fueron monotónicas. La amplitud se expresa en cuentas ADC, no en mV calibrados.

## Procesamiento comparado

Se compararon: señal sin filtrar; pasa-altas Butterworth IIR de cuarto orden a 20 y 40 Hz; pasa-banda IIR Butterworth de cuarto orden 20–400 Hz; y pasa-banda FIR de 101 coeficientes con ventana Hamming de 20–400 Hz. Los filtros se aplicaron en fase cero dentro de tramos válidos; gaps mayores de 1,5 ms se separaron sin interpolación. No se aplicó notch, porque el cribado espectral documentado no detectó una interferencia de red que superara el criterio preestablecido de 10 dB. Sin una referencia limpia no se puede declarar un filtro ganador ni medir SNR real.

Para la señal centrada de cada ventana, RMS = sqrt((1/N) sum x_c[n]^2). Se estimó la potencia espectral con periodograma unilateral Hann; MNF = sum(f_k P_k)/sum(P_k), y MDF es la frecuencia donde la potencia acumulada 20–400 Hz alcanza 50 %. La pendiente temporal se estimó por mínimos cuadrados y se resume por grupo con intervalos bootstrap a nivel de grupo. Una pendiente MDF/MNF negativa solo es compatible con el patrón clásico bajo contracciones sostenidas y fuerza aproximadamente controlada; aquí faltan fuerza/torque sincronizada, sitio anatómico confirmado, fase de repetición y referencia independiente. El primer tercio no se asume como basal, y no se fijó umbral ni inicio/pico de fatiga.

## Estructura

- `src/`: 18 módulos de QC, filtros, características, benchmark, análisis por pierna y resguardo de rutas locales.
- `src/cerqueira/`: cinco módulos separados para auditoría de alineación, extracción, LOSO e inferencia del objetivo de auto-reporte ordinal de miembro superior; sus productos deben mantenerse privados.
- `bibliografia/`: las 44 referencias consolidadas en dos archivos BibTeX, sin cambios.
- `datasets/README.md`: organización local segura; no incluye datos.
- reports/: informe IEEE público F31, agregados F55, estado F54, resumen y heatmap por pierna/serie, nota metodológica y auditoría.
- presentation/: guion hablado aprobado para la presentación Canva de 10 minutos.
- `manifest/SHA256SUMS`: hashes del paquete.

El manifiesto contiene los SHA-256 de los bytes versionados de cada archivo rastreado, excepto el propio manifiesto. Así los hashes permanecen estables aunque un checkout de Windows convierta finales de línea LF/CRLF.

## Reproducibilidad

Se verificaron Python 3.12.14, NumPy 2.3.5, pandas 3.0.1 y Pillow 12.3.0. El entorno mínimo y comandos de ayuda están en `requirements.txt` y en cada módulo. La inspección de ayuda no lee datos ni lanza entrenamiento. Los argumentos de datos y salida deben apuntar a almacenamiento autorizado fuera de este repositorio.

Los nombres de archivo derivados de manifiestos y tablas de etiquetas se validan como componentes individuales y las rutas se resuelven dentro de su raíz configurada, incluidas las comprobaciones de enlaces simbólicos. Los analizadores también confinan cada `output_path` del recibo a un archivo NPZ directo de la carpeta privada, comprueban su SHA-256 y validan el conjunto completo antes de crear productos. En las seis tablas CSV del modo agrupado, los identificadores con prefijo de fórmula se neutralizan solo al exportar; las claves originales se conservan durante el procesamiento.

El auditor Cerqueira también neutraliza en el CSV los textos con prefijo de fórmula procedentes de nombres de miembros ZIP; el recibo JSON conserva la ruta original para trazabilidad. El paquete sigue siendo intermedio y la auditoría formal de Codex Security no se ha cerrado.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m compileall -q src
python -m unittest discover -s tests -v
python src\entrenar_proxy_temporal.py --help
```

Verificación F50.6: la suite, `compileall`, la ayuda CLI, la demo sintética y la regresión real/fragmentada se registran en [el estado F50.6](reports/F50.6_STATUS_20261020.md). La ruta descriptiva no carga un modelo y declara abstención estructurada; conserva los resultados de señal y no produce puntuaciones ni niveles de fatiga. El ensayo real y sus recibos permanecen privados.

Antes de entrenar, inspecciona el esquema y las marcas temporales, define la etiqueta operacional, verifica el manifiesto de particiones por participante, mantén las salidas privadas fuera del repositorio y no trates el tercio temporal como fatiga o como fallo técnico.

### Ejecución algorítmica con señales sintéticas

`src/reproducir_emg_referencia.py` proporciona un recorrido ejecutable del procesamiento de señal sin grabaciones ni etiquetas del estudio. La demo genera fixtures de software deterministas (multisinusoides con ruido de baja amplitud, no una simulación fisiológica de EMG) para R1–R5, inserta un hueco temporal conocido y guarda sus productos fuera del checkout:

Cada `summary.json` exitoso incluye `supervised_inference`: el estado es `abstained` con motivo `no_supervised_model_configured`, `compatibility: not_evaluated` y `prediction_emitted: false`. El procesador completa y conserva los resultados descriptivos; no carga un clasificador ni emite niveles de fatiga. Esto no significa que un modelo haya detectado incompatibilidad: indica explícitamente que esta ruta no realiza inferencia supervisada.

```powershell
$demoRoot = Join-Path $env:TEMP 'emg-referencia-sintetico'
python src\reproducir_emg_referencia.py --demo --output-dir $demoRoot --overwrite
```

Puedes añadir `--dashboard` para crear un visor HTML autónomo con señal original/filtrada, RMS, MDF/MNF, FFT de una ventana seleccionable, PSD promedio por tramo, STFT y un Gantt de calidad y actividad candidata. El visor ofrece selección R1–R5, filtro e intervalo temporal. El umbral Otsu se calcula solo como segmentación exploratoria de ventanas RMS de 50 ms con 50 % de solapamiento sobre BP20–400. “Por encima” del umbral es una actividad candidata; “por debajo” no confirma reposo. Ninguna de estas marcas verifica repeticiones, artefactos, fatiga ni puntuaciones de modelo.

El modo `--input` admite dos esquemas, sin columnas de identidad. El esquema canónico contiene exactamente `series,time_s,adc_value`, donde `series` es R1–R5. El esquema de las grabaciones locales contiene `Relative_Time_s,ADC_Value` y acepta opcionalmente `Voltage_mV`; como no hay columna de serie, debe indicarse explícitamente `--series R1` (o R2–R5). El lector nunca infiere la serie desde el nombre del archivo. `Voltage_mV` se ignora porque no hay calibración física verificada. Para una grabación dividida en varios archivos, `--input-fragments` acepta la misma tabla de columnas y el mismo `--series`, ordena por los tiempos originales, rechaza duplicados o solapamientos y conserva los huecos sin interpolarlos:

```powershell
$privateOut = Join-Path $env:LOCALAPPDATA 'EMG_Biomecanica\emg_referencia'
python src\reproducir_emg_referencia.py --input '<CSV_WITH_Relative_Time_s_AND_ADC_Value>' --series R1 --output-dir $privateOut
```

```powershell
$rawFragments = @('<RAW_FRAGMENT_01.csv>', '<RAW_FRAGMENT_02.csv>')
python src\reproducir_emg_referencia.py --input-fragments $rawFragments --series R1 --output-dir $privateOut
```

El esquema canónico sigue siendo válido y no necesita `--series`:

```powershell
python src\reproducir_emg_referencia.py --input '<CSV_WITH_series_time_s_adc_value>' --output-dir $privateOut
```

Para generar el tablero de esta grabación, añade `--dashboard`. El HTML incrusta solo esa señal y sus resultados y debe permanecer en almacenamiento restringido. No contiene identificadores de participante ni resultados de inferencia. También escribe `activity_candidates.csv` con serie, tramo, inicio/fin/duración, umbral RMS, RMS medio/pico y estado de calidad común; sus filas tienen interpretación `candidate_activity_only`.

### Entrada tabular de grabaciones agrupadas

Hay un modo separado para quien ya disponga de un CSV de cohorte con IDs seudonimizados. Su contrato requiere exactamente las columnas `subject_id`, `side`, `series`, `time_s` y `adc_value`, en cualquier orden; `side` debe ser `left` o `right` y `series`, `R1`–`R5`. Cada combinación `(subject_id, side, series)` debe ocupar un solo bloque contiguo y sus marcas `time_s` deben aumentar estrictamente. El lector rechaza IDs vacíos o con espacios externos, valores de lateralidad/serie no válidos, bloques repetidos, marcas duplicadas o reinicios del reloj. Cada grupo debe producir al menos una ventana completa de 1 s después de aplicar la máscara común de bordes; si no, se detiene antes de escribir resultados.

```powershell
$privateOut = Join-Path $env:LOCALAPPDATA 'EMG_Biomecanica\emg_referencia_cohorte'
python src\reproducir_emg_referencia.py --cohort-input '<ANONYMIZED_COHORT_CSV>' --output-dir $privateOut
```

El tablero de una cohorte requiere elegir exactamente un grupo; sus identificadores se usan para seleccionar la entrada y no se incrustan en `dashboard.html`:

```powershell
python src\reproducir_emg_referencia.py --cohort-input '<ANONYMIZED_COHORT_CSV>' --dashboard --dashboard-subject 'P01' --dashboard-side left --dashboard-series R1 --output-dir $privateOut
```

El modo procesa cada grupo de forma independiente y conserva `subject_id` y `side` en las tablas de señal y ventana para mantener la clave `(subject_id, side, series)`. Además genera tres resúmenes agregados sin identificadores: `cohort_by_side_series.csv`, `cohort_r1_r5_paired.csv` y `cohort_bilateral_paired.csv`. Primero reduce las ventanas solapadas a una mediana por grabación; después calcula medianas y rangos intercuartílicos entre participantes. R1–R5 usa únicamente pares completos del mismo participante y lado; la comparación bilateral usa pares completos de la misma persona y serie. El cambio de RMS R1–R5 se expresa como porcentaje relativo a R1; el contraste bilateral de RMS se expresa en cuentas ADC; MDF/MNF se expresan en Hz. Estos resúmenes son descriptivos y no generan etiquetas de fatiga ni inferencia fisiológica validada. `features.csv` no contiene la máscara de actividad F39, por lo que sus estadísticas espectrales no reproducen el análisis local con máscara.

Las salidas de señal/ventana y los tres resúmenes de cohorte deben permanecer en almacenamiento restringido. La herramienta no puede verificar que un pseudónimo sea irreidentificable ni que las etiquetas coincidan con la identidad real de la grabación; esas claves deben confirmarse antes de preparar el CSV. La lectura individual también requiere timestamps crecientes, rechaza filas incompletas/no finitas y conserva discontinuidades temporales para el analizador. El adaptador de CSV crudo se verificó con un archivo completo. En una comprobación separada, `--cohort-fragments` reensambló CSV agrupados con claves explícitas y sus siete tablas y dashboard coincidieron con el modo de CSV agrupado completo. La opción adicional `--input-fragments` procesa fragmentos crudos sin añadir claves de cohorte. El recibo, la señal y las salidas detalladas están en almacenamiento privado y no se publican. Estas son verificaciones de procesamiento descriptivo, no una validación de la máscara fisiológica F39, una inferencia de modelo ni una detección de fatiga.

### Reensamblado de fragmentos

Para una grabación individual en el esquema original `Relative_Time_s,ADC_Value[,Voltage_mV]`, usa `--input-fragments` y una serie explícita. No hace falta agregar identificadores, lateralidad u otras columnas. Los fragmentos se reensamblan en memoria por marca temporal; los solapamientos y timestamps duplicados se rechazan, y los huecos se conservan sin interpolación. `summary.json` registra el número de archivos, muestras y huecos detectados. Esto es procesamiento por lotes, no streaming con estado; si cada exportación reinicia el reloj, el sistema no estima un desfase.

```powershell
$rawFragments = @('<RAW_FRAGMENT_01.csv>', '<RAW_FRAGMENT_02.csv>', '<RAW_FRAGMENT_03.csv>')
$privateOut = Join-Path $env:LOCALAPPDATA 'EMG_Biomecanica\emg_fragmentos_raw'
python src\reproducir_emg_referencia.py --input-fragments $rawFragments --series R1 --dashboard --output-dir $privateOut
```

Para datos de cohorte, usa `--cohort-fragments` con el esquema agrupado de cinco columnas (`subject_id,side,series,time_s,adc_value`). Cada fragmento conserva las marcas temporales originales y recibe la misma clave explícita `(subject_id, side, series)`; el programa no infiere identidad, lado o serie de los nombres de archivo. Ordena los fragmentos por marca de tiempo dentro de cada clave, rechaza solapamientos, timestamps duplicados y reinicios del reloj, y conserva los huecos sin interpolar. Este esquema agrupado es independiente del modo de grabación individual y se usa solo cuando las claves explícitas ya están disponibles.

```powershell
$fragments = @('<FRAGMENT_01.csv>', '<FRAGMENT_02.csv>', '<FRAGMENT_03.csv>')
$privateOut = Join-Path $env:LOCALAPPDATA 'EMG_Biomecanica\emg_fragmentos_reensamblados'
python src\reproducir_emg_referencia.py --cohort-fragments $fragments --output-dir $privateOut
```

Se puede añadir `--dashboard` junto con `--dashboard-subject`, `--dashboard-side` y `--dashboard-series` para visualizar un único grupo reensamblado. La salida se calcula después de ordenar los fragmentos por sus marcas de tiempo originales.

Ambas rutas de fragmentos recomponen la señal en memoria antes de filtrar y calcular características. Con fragmentos contiguos y marcas de tiempo intactas, las tablas de análisis coinciden con las de la grabación completa; los huecos reales permanecen separados. La prueba de equivalencia no representa una inferencia de modelo ni una reconstrucción de predicciones/Gantt.

Para entradas de señales reales, conserva el CSV y todas las salidas en almacenamiento local autorizado y restringido. La herramienta impide escribir dentro del checkout, pero no detecta carpetas sincronizadas por otros servicios. No publiques los resultados a nivel de ventana o serie.

El flujo estima la frecuencia por la mediana de los intervalos temporales, separa gaps mayores que `max(1,5 ms, 1,5 × mediana del intervalo) + 1 ns` sin interpolar y omite tramos con dispersión temporal p95–p5 superior al 5 %. Requiere frecuencia estimada mayor de 800 Hz para la rama de 20–400 Hz y no remuestrea. Compara señal de control, HP20, HP40 y BP20–400 Butterworth de cuarto orden en fase cero, y BP20–400 FIR Hamming de 101 coeficientes. Una máscara común excluye los bordes de todos los filtros; sobre muestras válidas calcula ventanas Hann de 1 s con 50 % de solapamiento, RMS, MDF y MNF entre 20 y 400 Hz, FFT unilateral, PSD promedio y STFT. El archivo `features.csv` incluye tercios de posición temporal (`temprana`, `transicion`, `tardia`), que son un proxy de tiempo y no clases de fatiga.

La salida contiene `segments.csv` (un resumen por tramo), `mask.csv` (máscara de validez por muestra y rama), `features.csv`, `fft.csv`, `psd.csv`, `stft.csv`, `activity_candidates.csv` (vacío salvo cuando se solicita el tablero) y un `summary.json` sin rutas ni hashes de entrada. El modo agrupado añade los tres CSV agregados descritos arriba. Con `--dashboard` también se genera `dashboard.html` para una grabación sin identidad o para un único grupo seleccionado. El Gantt distingue actividad candidata por envolvente RMS de muestras no evaluables por la máscara de procesamiento; no clasifica calidad fisiológica, movimiento o saturación. `input_source` identifica `synthetic_demo`, `csv_input`, `csv_fragments`, `cohort_csv` o `cohort_fragments` según la entrada; el CLI no verifica si ese archivo es sintético ni si su uso está autorizado. `--overwrite` permite reemplazar archivos de salida conocidos en una carpeta existente y elimina tablas agregadas de una ejecución de cohorte anterior cuando la nueva entrada no es agrupada, además de retirar un tablero anterior si no se solicita uno; así evita dejar productos opcionales obsoletos junto a la nueva salida. Las métricas numéricas vacías o no finitas pueden quedar ausentes por métrica, pero cualquier texto no vacío que no sea numérico detiene el resumen para no alterar silenciosamente sus denominadores. El flujo no incluye notch, estimación de fuerza, etiquetas fisiológicas, entrenamiento, predicciones, modelo serializado ni puntuaciones de fatiga. La ejecución sintética hace reproducible el contrato algorítmico, pero **no reproduce las métricas, predicciones ni resultados F31 de la cohorte local**. No se añadieron dependencias: NumPy ya está fijado en `requirements.txt`; el filtro se comparte con el módulo público `src/comparar_filtros_emg.py`.

Este recorrido prueba el comportamiento numérico del software. No verifica que una entrada sea EMG fisiológica, no identifica RAW/RECT/ENV o una calibración física y no selecciona un filtro fisiológicamente ganador.

### Código reproducible Cerqueira (fatiga autopercibida de miembro superior)

El paquete ahora incluye el código de auditoría de alineación EMG-etiquetas, extracción de características, evaluación anidada LOSO por participante e inferencia de ensayos/fragmentos compatibles en [`src/cerqueira/`](src/cerqueira/README.md). Sus entradas y salidas granulares contienen información por participante, ensayo o ventana: mantenlas en almacenamiento restringido fuera del checkout y no las publiques. Los tests incluidos usan datos sintéticos.

La inferencia admite dos manifiestos sin consumir etiquetas: `--fold-manifest` para un modelo LOSO incluido en el manifiesto de pliegues y `--final-manifest` para un artefacto full-fit entregado localmente por el usuario. En ambos casos comprueba el SHA-256, la familia y dimensiones del modelo, el orden de las características y el contrato de preprocesamiento y extracción. El extractor admite ventanas completas contiguas sin solapamiento; los manifiestos con `window_overlap_fraction` distinto de cero se rechazan explícitamente porque esa configuración no está implementada. El paquete **no incluye ni distribuye** un checkpoint final; los modelos y las entradas/salidas de inferencia deben permanecer fuera del repositorio en almacenamiento autorizado. El objetivo sigue siendo el auto-reporte ordinal de fatiga autopercibida de miembro superior; no es detección validada de fatiga funcional, fatiga del cuádriceps ni una predicción clínica.

Ejemplo para un artefacto full-fit compatible disponible localmente (sustituye las rutas por rutas autorizadas; las salidas deben quedar fuera del checkout):

```powershell
python src\cerqueira\inferir_cerqueira_trial.py '<COMPATIBLE_TRIAL.csv>' `
  --model '<PRIVATE_FINAL_MODEL.npz>' `
  --final-manifest '<PRIVATE_FINAL_MODEL_MANIFEST.json>' `
  --output-csv '<PRIVATE_OUTPUT\scores.csv>' `
  --summary-json '<PRIVATE_OUTPUT\summary.json>'
```

La forma LOSO existente conserva `--fold-manifest`; ambos argumentos de manifiesto son mutuamente excluyentes. Los archivos mencionados en este ejemplo no forman parte del paquete público.

### Componentes que aún requieren datos o validación autorizados

El repositorio también contiene una pantalla exploratoria de cruces sostenidos de RMS/MDF (`src/analizar_umbral_senal_emg.py`), pero su ejecución completa lee un manifiesto, el recibo de sensibilidad y señales/arrays privados. Sus salidas pueden incluir identificadores, tiempos por registro y figuras individuales; mantenlas en almacenamiento restringido. La pantalla no cuenta repeticiones ni demuestra fatiga o fallo técnico. La tabla de componentes, alcance reproducible y prerrequisitos está en `reports/PUBLIC_RELEASE_AUDIT.md`.

No hay entradas ni evaluación de Borg/RPE en el paquete. Tampoco se publica un checkpoint ni un comando de inferencia listo para uso: el código existente entrena y evalúa el proxy de posición temporal con etiquetas, particiones y trazas privadas. No uses esas clases como etiquetas fisiológicas. Reproducir el benchmark F31 exige los archivos locales autorizados y sus recibos coincidentes; el demo sintético no reproduce sus métricas.

## Límites y uso

La guía prescribe salida RAW y muestreo mínimo de 1 kHz por canal, además de ubicación de electrodos sobre recto femoral y ejercicio unilateral al 70 % de 1RM; esto define el protocolo objetivo, no certifica la adquisición de cada CSV. El intervalo mediano de timestamps es cercano a 1 ms (aproximadamente 1 kHz nominal). En el archivo no se verifican por registro del sitio real del electrodo, carga, pausas, cadencia o repeticiones mecánicas. `Voltage_mV` reproduce una transformación aritmética de `ADC_Value`, la conclusión correcta es que el estudio se dirige a un grupo muscular y ejercicio específicos, mientras los resultados disponibles describen tendencias del canal guardado.

