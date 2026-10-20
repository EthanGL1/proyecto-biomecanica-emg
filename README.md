# Análisis EMG — paquete público seguro F50 (intermedio)

**Actualización intermedia del 20 de octubre de 2026.** Este paquete reúne código de procesamiento, pruebas, bibliografía y resultados agregados permitidos. El informe DOCX/PDF F31 que permanece en `reports/` es histórico y corresponde a un benchmark de posición temporal; no es el manuscrito LaTeX F49 ni una versión final del estudio.

El [corte de estado F50](reports/F50_STATUS_20261020.md) detalla las verificaciones actuales, el alcance de cada rama y los bloqueos que siguen abiertos.

El análisis fisiológico principal usa las grabaciones locales de cuádriceps. No hay etiquetas independientes de fatiga funcional, fuerza/torque sincronizados, repeticiones verificadas ni calibración física del canal, por lo que no se presenta un detector local de fatiga. El benchmark Borg es un experimento separado de esfuerzo percibido en EMG de hombro. Cerqueira es una tercera rama para auto-reporte ordinal de fatiga de miembro superior: corpus, LOSO y una inferencia retenida están verificados localmente, pero el baseline temporal supera al predictor EMG y no se recomienda un modelo EMG para despliegue.

Este paquete público no contiene señales, HTML con señales embebidas, resultados por participante/ventana, datos de identificación, checkpoints ni inferencias individuales. El modelo Borg congelado no se publica: el SHA está auditado, pero la carga e inferencia reproducibles siguen sin demostrarse en un runtime autorizado.

## Resultados consolidados F50

El inventario local contiene 140 grabaciones de 15 grupos derivados de nombres de archivo; 138 registros formales R1–R5 sustentan el análisis descriptivo y dos registros R0/R0_1 se conservaron solo para control de calidad. Hay 27 pares emparejables de participante y pierna para R1 frente a R5. En esos pares, el cambio mediano fue RMS +31,27 % (IC del 95 %: 5,11 a 40,04), MDF +1,45 Hz (−4,49 a 2,22) y MNF −2,06 Hz (−6,56 a 1,00). Los intervalos espectrales incluyen cero: no se observó una reducción de MDF/MNF consistente en el conjunto. Estos son cambios eléctricos descriptivos, no niveles ni eventos confirmados de fatiga.

Se auditaron 2.102 candidatos de activación; no están sincronizados con marcadores mecánicos y no se denominan repeticiones verificadas. La evaluación visual de una muestra tampoco proporciona una referencia independiente para sensibilidad o especificidad de eventos.

El benchmark separado de Borg incluye 34 participantes y 1.309 ventanas OOF para RPE ≥17. El procedimiento de selección anidada agrupada por participante obtuvo macro-F1 agrupado 0,6176 y macro-F1 medio por participante 0,5787 (DE 0,1344). Mide esfuerzo percibido alto en el protocolo de hombro, no fatiga fisiológica de cuádriceps. El checkpoint Borg no ha producido aún una predicción reproducible.

La rama Cerqueira conserva su objetivo como fatiga autopercibida ordinal de miembro superior (0/1/2). Los ZIP oficiales de señal y etiquetas coinciden con los checksums del registro Zenodo; se emparejaron 13 participantes y 156 ensayos, con 34 ensayos que contienen marcas no monótonas y 9 ensayos/11 filas con discordancias entre canales. El LOSO primario (≥90 % de cobertura por ensayo) incluye 146 ensayos y 44.886 ventanas; la sensibilidad ≥95 % incluye 143 ensayos y 44.428 ventanas. Macro-F1 medio por participante fue 0,368 (IC95 % 0,319–0,415) para EMG frente a 0,713 (0,666–0,755) para el baseline temporal. Las etiquetas son fuertemente crecientes con el tiempo (Spearman medio por ensayo 0,9022). El checkpoint de pliegue retenido produjo 223 predicciones reproducibles sobre un ensayo real sin usar etiquetas. Es validación interna dentro del mismo dataset; no es una prueba externa ni un detector de pérdida funcional o de fatiga de cuádriceps. Los datos, artefactos y predicciones granulares permanecen restringidos y no forman parte de este repositorio.

La edición F48 del manuscrito conserva los 44 registros bibliográficos y la estructura del template docente. El validador estructural confirma las secciones requeridas, citas, tablas y figuras; la compilación IEEE sigue pendiente porque no está disponible la distribución Tectonic requerida y no hay motor TeX local verificado. No se declara un PDF ni una extensión de páginas como completados.

## Alcance y datos

El análisis de señales EMG, la extracción de características y la evaluación del proxy temporal usaron las 140 grabaciones EMG correspondientes a 15 grupos derivados de los códigos de archivo. Los 138 registros R1–R5 participaron en el benchmark de tercios temporales; R0/R0_1 se conservaron para control de calidad. `participants.csv` se utilizó para reconciliación administrativa del proyecto, pero sus campos no fueron entradas de este análisis ni covariables del modelo; no se emplearon peso, altura u otros metadatos demográficos. Las particiones mantuvieron todas las grabaciones de cada persona en un mismo grupo.

No se publican señales, datos por participante, características, etiquetas, particiones, predicciones, resultados por pliegue ni recibos privados. Las mediciones originales y salidas con granularidad individual permanecen en el almacenamiento local restringido. El benchmark completo requiere acceso autorizado a los datos. La comprobación exploratoria externa posterior al congelamiento del modelo no se incluye como datos o figura en esta edición pública.

## Resultados históricos agregados locales (F31; clasificación de posición temporal)

Los valores siguientes pertenecen a la publicación histórica F31. El análisis F48 posterior añade el filtro de actividad, el análisis de duraciones/ocupaciones y las comparaciones emparejadas R1–R5 descritas arriba; los resultados F31 no deben interpretarse como el cierre fisiológico actual.

La selección anidada LOSO comparó 325 configuraciones de 11 familias con 15 pliegues externos por participante y cuatro pliegues internos agrupados. Para los tres tercios temporales, el procedimiento anidado obtuvo macro-F1 medio por participante 0,740 (IC 95 % bootstrap 0,689–0,787) y ROC-AUC macro OvR 0,941 (0,921–0,959). La regresión logística tuvo el mayor punto de macro-F1 entre selectores de familia fija (0,751), pero no se estableció superioridad ni un único modelo universal.

La estratificación post hoc separó las piernas registradas. La izquierda incluyó 14 grupos, 70 grabaciones y 210 segmentos; la derecha, 15 grupos, 68 grabaciones y 204 segmentos. Catorce grupos fueron pareados. Macro-F1 fue 0,776 (IC 95 % 0,675–0,874) a izquierda y 0,686 (0,611–0,762) a derecha; la diferencia pareada izquierda menos derecha fue +0,092 (−0,052 a +0,238), que incluye cero. La cobertura por serie se resume en `reports/LATERALITY_F31_20261018.md` y el heatmap agregado. Estas métricas describen clasificación de posición temporal, no fatiga ni asimetría fisiológica.

En ventanas de 1 s con 50 % de solapamiento, la rama HP20 dio pendientes medianas por grupo de +19,30 cuentas ADC/min para RMS (IC95 % +12,06 a +41,41), −52,49 Hz/min para MDF (−68,38 a −8,54) y −44,03 Hz/min para MNF (−86,01 a −22,17). Las pendientes espectrales mantuvieron dirección descendente al repetir el cálculo sin solapamiento y en las otras ramas. La linealidad fue baja (R² mediano: RMS 0,031; MDF 0,118; MNF 0,083) y las trayectorias no fueron monotónicas. La amplitud se expresa en cuentas ADC, no en mV calibrados.

## Procesamiento comparado

Se compararon: señal sin filtrar; pasa-altas Butterworth IIR de cuarto orden a 20 y 40 Hz; pasa-banda IIR Butterworth de cuarto orden 20–400 Hz; y pasa-banda FIR de 101 coeficientes con ventana Hamming de 20–400 Hz. Los filtros se aplicaron en fase cero dentro de tramos válidos; gaps mayores de 1,5 ms se separaron sin interpolación. No se aplicó notch, porque el cribado espectral documentado no detectó una interferencia de red que superara el criterio preestablecido de 10 dB. Sin una referencia limpia no se puede declarar un filtro ganador ni medir SNR real.

Para la señal centrada de cada ventana, RMS = sqrt((1/N) sum x_c[n]^2). Se estimó la potencia espectral con periodograma unilateral Hann; MNF = sum(f_k P_k)/sum(P_k), y MDF es la frecuencia donde la potencia acumulada 20–400 Hz alcanza 50 %. La pendiente temporal se estimó por mínimos cuadrados y se resume por grupo con intervalos bootstrap a nivel de grupo. Una pendiente MDF/MNF negativa solo es compatible con el patrón clásico bajo contracciones sostenidas y fuerza aproximadamente controlada; aquí faltan fuerza/torque sincronizada, sitio anatómico confirmado, fase de repetición y referencia independiente. El primer tercio no se asume como basal, y no se fijó umbral ni inicio/pico de fatiga.

## Estructura

- `src/`: 18 módulos de QC, filtros, características, benchmark, análisis por pierna y resguardo de rutas locales.
- `bibliografia/`: las 44 referencias consolidadas en dos archivos BibTeX, sin cambios.
- `datasets/README.md`: organización local segura; no incluye datos.
- `reports/`: informe IEEE público F31, resumen y heatmap agregado por pierna/serie, nota metodológica y auditoría del paquete.
- `manifest/SHA256SUMS`: hashes del paquete.

El manifiesto contiene los SHA-256 de los bytes versionados de cada archivo rastreado, excepto el propio manifiesto. Así los hashes permanecen estables aunque un checkout de Windows convierta finales de línea LF/CRLF.

## Reproducibilidad

Se verificaron Python 3.12.14, NumPy 2.3.5, pandas 3.0.1 y Pillow 12.3.0. El entorno mínimo y comandos de ayuda están en `requirements.txt` y en cada módulo. La inspección de ayuda no lee datos ni lanza entrenamiento. Los argumentos de datos y salida deben apuntar a almacenamiento autorizado fuera de este repositorio.

Los nombres de archivo derivados de manifiestos y tablas de etiquetas se validan como componentes individuales y las rutas se resuelven dentro de su raíz configurada, incluidas las comprobaciones de enlaces simbólicos. Los analizadores también confinan cada `output_path` del recibo a un archivo NPZ directo de la carpeta privada, comprueban su SHA-256 y validan el conjunto completo antes de crear productos. En las seis tablas CSV del modo agrupado, los identificadores con prefijo de fórmula se neutralizan solo al exportar; las claves originales se conservan durante el procesamiento.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m compileall -q src
python -m unittest discover -s tests -v
python src\entrenar_proxy_temporal.py --help
```

Antes de entrenar, inspecciona el esquema y las marcas temporales, define la etiqueta operacional, verifica el manifiesto de particiones por participante, mantén las salidas privadas fuera del repositorio y no trates el tercio temporal como fatiga o como fallo técnico.

### Ejecución algorítmica independiente con señales sintéticas

`src/reproducir_emg_referencia.py` proporciona un recorrido ejecutable del procesamiento de señal sin grabaciones ni etiquetas del estudio. La demo genera fixtures de software deterministas (multisinusoides con ruido de baja amplitud, no una simulación fisiológica de EMG) para R1–R5, inserta un hueco temporal conocido y guarda sus productos fuera del checkout:

```powershell
$demoRoot = Join-Path $env:TEMP 'emg-referencia-sintetico'
python src\reproducir_emg_referencia.py --demo --output-dir $demoRoot --overwrite
```

Puedes añadir `--dashboard` para crear un visor HTML autónomo con señal original/filtrada, RMS, MDF/MNF, FFT de una ventana seleccionable, PSD promedio por tramo, STFT y un Gantt de calidad y actividad candidata. El visor ofrece selección R1–R5, filtro e intervalo temporal. El umbral Otsu se calcula solo como segmentación exploratoria de ventanas RMS de 50 ms con 50 % de solapamiento sobre BP20–400. “Por encima” del umbral es una actividad candidata; “por debajo” no confirma reposo. Ninguna de estas marcas verifica repeticiones, artefactos, fatiga ni puntuaciones de modelo.

También acepta un CSV sintético o autorizado que no contenga columnas de identidad y tenga exactamente tres columnas: `series,time_s,adc_value`. Cada valor de `series` debe ser R1, R2, R3, R4 o R5; cada serie debe aparecer en un único bloque contiguo del archivo. Ejecuta un archivo por grabación. El lector rechaza columnas adicionales para impedir que entren IDs de participante o metadatos no previstos:

```powershell
$privateOut = Join-Path $env:LOCALAPPDATA 'EMG_Biomecanica\emg_referencia'
python src\reproducir_emg_referencia.py --input '<CSV_WITHOUT_ID_COLUMNS>' --output-dir $privateOut
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

Las salidas de señal/ventana incluyen pseudónimo y deben permanecer en almacenamiento restringido. Los tres archivos agregados no incluyen IDs directos, pero sus conteos pueden ser pequeños y también deben tratarse como restringidos; agregación no equivale a anonimización. La herramienta no puede verificar que un pseudónimo sea irreidentificable, que las etiquetas coincidan con la identidad real de la grabación o que el CSV cumpla metadatos de adquisición. Como el contrato no tiene un identificador de grabación independiente, no puede detectar si una nueva combinación de claves representa una etiqueta equivocada; debe confirmarse antes de preparar el CSV. Se ejecutó una prueba de humo previa con un único registro formal real, cuyo SHA-256 se enlazó con el manifiesto privado; esto comprueba el lector y procesador para ese registro, pero no valida un lote completo de 138 grabaciones ni todas las 15 personas. La entrada de una sola señal `--input` conserva su esquema estricto de tres columnas y no acepta IDs.

### Reensamblado de fragmentos contiguos

Para procesar una grabación que se exportó en varios CSV, usa `--cohort-fragments` con el mismo esquema agrupado. Cada archivo debe conservar las marcas de tiempo originales y una combinación `(subject_id, side, series)` por bloque. El programa ordena los fragmentos por marca de tiempo dentro de cada clave, rechaza solapamientos, marcas duplicadas y reinicios de reloj, y conserva los huecos para que el analizador separe los tramos sin interpolar.

```powershell
$fragments = @('<FRAGMENT_01.csv>', '<FRAGMENT_02.csv>', '<FRAGMENT_03.csv>')
$privateOut = Join-Path $env:LOCALAPPDATA 'EMG_Biomecanica\emg_fragmentos_reensamblados'
python src\reproducir_emg_referencia.py --cohort-fragments $fragments --output-dir $privateOut
```

Se puede añadir `--dashboard` junto con `--dashboard-subject`, `--dashboard-side` y `--dashboard-series` para visualizar un único grupo reensamblado. La salida se calcula después de ordenar los fragmentos por sus marcas de tiempo originales.

Este modo recompone la señal en memoria antes de filtrar y calcular características. Es procesamiento por lotes fuera de línea, no streaming con estado: si los timestamps se reinician en cada exportación, no se adivinan desplazamientos temporales y la entrada se rechaza si causa solapamientos. Con fragmentos contiguos y marcas de tiempo intactas, las tablas de análisis coinciden con las de la grabación completa; los huecos reales permanecen separados. La prueba de equivalencia no representa una inferencia de modelo ni una reconstrucción de predicciones/Gantt.

Para entradas de señales reales, conserva el CSV y todas las salidas en almacenamiento local autorizado y restringido. La herramienta impide escribir dentro del checkout, pero no detecta carpetas sincronizadas por otros servicios. No publiques los resultados a nivel de ventana o serie.

El flujo estima la frecuencia por la mediana de los intervalos temporales, separa gaps mayores que `max(1,5 ms, 1,5 × mediana del intervalo) + 1 ns` sin interpolar y omite tramos con dispersión temporal p95–p5 superior al 5 %. Requiere frecuencia estimada mayor de 800 Hz para la rama de 20–400 Hz y no remuestrea. Compara señal de control, HP20, HP40 y BP20–400 Butterworth de cuarto orden en fase cero, y BP20–400 FIR Hamming de 101 coeficientes. Una máscara común excluye los bordes de todos los filtros; sobre muestras válidas calcula ventanas Hann de 1 s con 50 % de solapamiento, RMS, MDF y MNF entre 20 y 400 Hz, FFT unilateral, PSD promedio y STFT. El archivo `features.csv` incluye tercios de posición temporal (`temprana`, `transicion`, `tardia`), que son un proxy de tiempo y no clases de fatiga.

La salida contiene `segments.csv` (un resumen por tramo), `mask.csv` (máscara de validez por muestra y rama), `features.csv`, `fft.csv`, `psd.csv`, `stft.csv`, `activity_candidates.csv` (vacío salvo cuando se solicita el tablero) y un `summary.json` sin rutas ni hashes de entrada. El modo agrupado añade los tres CSV agregados descritos arriba. Con `--dashboard` también se genera `dashboard.html` para una grabación sin identidad o para un único grupo seleccionado. El Gantt distingue actividad candidata por envolvente RMS de muestras no evaluables por la máscara de procesamiento; no clasifica calidad fisiológica, movimiento o saturación. `input_source` identifica `synthetic_demo`, `csv_input`, `cohort_csv` o `cohort_fragments` según la entrada; el CLI no verifica si ese archivo es sintético ni si su uso está autorizado. `--overwrite` permite reemplazar archivos de salida conocidos en una carpeta existente y elimina tablas agregadas de una ejecución de cohorte anterior cuando la nueva entrada no es agrupada, además de retirar un tablero anterior si no se solicita uno; así evita dejar productos opcionales obsoletos junto a la nueva salida. Las métricas numéricas vacías o no finitas pueden quedar ausentes por métrica, pero cualquier texto no vacío que no sea numérico detiene el resumen para no alterar silenciosamente sus denominadores. El flujo no incluye notch, estimación de fuerza, etiquetas fisiológicas, entrenamiento, predicciones, modelo serializado ni puntuaciones de fatiga. La ejecución sintética hace reproducible el contrato algorítmico, pero **no reproduce las métricas, predicciones ni resultados F31 de la cohorte local**. No se añadieron dependencias: NumPy ya está fijado en `requirements.txt`; el filtro se comparte con el módulo público `src/comparar_filtros_emg.py`.

Este recorrido prueba el comportamiento numérico del software. No verifica que una entrada sea EMG fisiológica, no identifica RAW/RECT/ENV o una calibración física y no selecciona un filtro fisiológicamente ganador.

### Componentes que aún requieren datos o validación autorizados

El repositorio también contiene una pantalla exploratoria de cruces sostenidos de RMS/MDF (`src/analizar_umbral_senal_emg.py`), pero su ejecución completa lee un manifiesto, el recibo de sensibilidad y señales/arrays privados. Sus salidas pueden incluir identificadores, tiempos por registro y figuras individuales; mantenlas en almacenamiento restringido. La pantalla no cuenta repeticiones ni demuestra fatiga o fallo técnico. La tabla de componentes, alcance reproducible y prerrequisitos está en `reports/PUBLIC_RELEASE_AUDIT.md`.

No hay entradas ni evaluación de Borg/RPE en el paquete. Tampoco se publica un checkpoint ni un comando de inferencia listo para uso: el código existente entrena y evalúa el proxy de posición temporal con etiquetas, particiones y trazas privadas. No uses esas clases como etiquetas fisiológicas. Reproducir el benchmark F31 exige los archivos locales autorizados y sus recibos coincidentes; el demo sintético no reproduce sus métricas.

## Límites y uso

No se verificaron modalidad RAW/RECT/ENV, frecuencia física del equipo, calibración, sitio muscular, carga, pausas, cadencia o repeticiones mecánicas. La columna local `Voltage_mV` no está validada como voltaje físico. Por tanto, esta investigación describe tendencias del canal guardado y desempeño de una tarea proxy, no impacto de un grupo muscular específico ni fatiga confirmada.

No se agregó una licencia: no se han verificado los derechos para otorgarla. El acceso a datos sigue sujeto a autorización, consentimiento y aprobación ética aplicable.
