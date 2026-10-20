# Auditoría del paquete público EMG — F31/F44

**Fecha:** 19–20 de octubre de 2026
**Estado:** candidato público intermedio. La base remota revisada antes de esta ampliación fue `9a53053f819580803fe01f148429abc3ba7b5731`; esta actualización incorpora una interfaz CSV agrupada, reensamblado de fragmentos y correcciones puntuales de seguridad. El paquete sigue sin incluir un modelo Borg verificable ni una validación fisiológica de fatiga.

**Referencia remota base:** `main` apuntaba a `9a53053f819580803fe01f148429abc3ba7b5731` (`Fix public EMG pipeline provenance and audit metadata`) antes de esta actualización. La matriz describe también los límites de la interfaz agrupada añadida en F43; la verificación del commit remoto actualizado se registra en la gestión local del proyecto.

## Contenido y exclusiones

El paquete contiene 17 módulos Python, dos bibliografías consolidadas, instrucciones, el informe IEEE público F31 en DOCX/PDF, el heatmap y el resumen agregado por pierna/serie. No contiene archivos de participantes, señales del estudio locales o externas, métricas por persona/pliegue/canal, etiquetas, características, particiones, predicciones, recibos privados, modelos serializados ni figuras de caso. `participants.csv` se excluyó del paquete; se utilizó para reconciliación administrativa del proyecto, pero no fue entrada del análisis de señales/características ni fuente de covariables. El CLI de referencia crea datos sintéticos solo cuando se ejecuta.

El informe público conserva resultados agregados locales. Se eliminaron las figuras de R0/R0_1, los puntos de rendimiento por participante y la señal de una grabación individual; también se borraron del DOCX los medios incrustados sin uso. Las 44 referencias consolidadas permanecen sin cambios.

## Controles verificados

- Los módulos compilan; los módulos de lateralidad, filtros y referencia rechazan rutas de entrada o salida dentro del checkout público. Los flujos de rutas derivadas comprueban su contención y los identificadores de cohorte se neutralizan en la frontera de exportación CSV.
- La suite local pasó 43 pruebas de `unittest`, incluidas pruebas de traversal POSIX/Windows, junctions, pseudónimos Unicode y neutralización de fórmulas en las seis tablas. `pip check` no encontró dependencias rotas y `compileall` terminó correctamente. `--help` no lee grabaciones ni entrena.
- `src/reproducir_emg_referencia.py --demo` procesa fixtures de software multisinusoidales R1–R5 (no un simulador fisiológico de EMG) y deja outputs temporales fuera del checkout; no genera resultados ni métricas de la cohorte F31.
- DOCX con 44 referencias, 15 tablas nativas y ocho figuras agregadas o esquemas editables; sin checklist de envío, placeholders ni contenido individual de las tres figuras retiradas.
- PDF exportado desde Word en A4, con 22 páginas renderizadas e inspeccionadas; no se detectaron desbordes ni tablas rotas.
- Las dos bibliografías se cotejaron con las versiones consolidadas y no se modificaron.
- La matriz por lado reporta 14 grupos izquierdos y 15 derechos, con 14 pareados; la diferencia Macro-F1 incluye cero.
- El manifiesto `SHA256SUMS` de la base histórica tenía 33 entradas; se recalculará y comprobará para esta revisión antes de cualquier actualización remota. La actualización F44-04 aún debe reconciliarse con el estado actual de `main`.

## Matriz de brechas de reproducibilidad

| Componente | Código y evidencia disponible | Límite de reproducción pública |
|---|---|---|
| Filtrado y calidad de señal | `src/comparar_filtros_emg.py` separa gaps, diseña/aplica filtros IIR y FIR, y calcula máscaras de bordes; `src/reproducir_emg_referencia.py` ejecuta cinco ramas con la fixture sintética, el CSV individual de tres columnas sin identidad o el modo separado `--cohort-input` con `subject_id,side,series,time_s,adc_value`. Este último valida bloques únicos, lateralidad, serie, orden temporal y una ventana válida por grupo; etiqueta las filas de salida y no calcula comparaciones de cohorte. Se corrió un smoke de un registro real enlazado por SHA-256 al manifiesto privado. | Reproduce el algoritmo numérico y pasó una prueba de humo para un registro local. Esto no verifica la validez fisiológica de la modalidad RAW/RECT/ENV, la calibración, el sitio muscular, los artefactos o la calidad de movimiento, y no demuestra el procesamiento de los 138 registros en modo cohorte. No hay referencia limpia para medir SNR o declarar un filtro ganador. |
| FFT, PSD y STFT | La herramienta de referencia exporta `fft.csv`, `psd.csv` y `stft.csv`; `tests/test_public_emg_pipeline.py` comprueba frecuencia/potencia y las filas generadas. | Reproducible con datos sintéticos o CSV autorizado sin identidad. No reproduce por sí solo el espectro agregado de la cohorte. |
| RMS, MDF y MNF | La misma herramienta calcula estas características sobre ventanas de 1 s con 50 % de solapamiento y banda 20–400 Hz; la prueba sintética comprueba valores esperados. | Unidades ADC, sin calibración física. Las tendencias de cohorte de F31 publicadas son resultados agregados, no se pueden recalcular desde los archivos de este repositorio. |
| Series R1–R5 | El lector público acepta únicamente las etiquetas R1, R2, R3, R4 y R5; la fixture de demo contiene las cinco. | Las etiquetas son nombres de series. No codifican repeticiones, fases de ejercicio ni sucesos mecánicos verificados. No se incluyen grabaciones del estudio. |
| Eventos o umbrales | `src/analizar_umbral_senal_emg.py` implementa ventanas válidas, umbrales robustos y cruces sostenidos candidatos de RMS/MDF. El CLI completo requiere `manifest_pseudonymized.csv`, `receipt_filter_sensitivity_v1.json`, señales pseudonimizadas y archivos filtrados; escribe filas por registro/ventana y figuras individuales en su salida privada. | No se ejecuta desde la demo pública ni con los archivos publicados. Los cruces son descensos exploratorios respecto a ventanas tempranas; no detectan repeticiones, inicio/pico de fatiga, fallo técnico ni eventos validados frente a vídeo, fuerza o anotación independiente. |
| Sensibilidad a filtros | Las funciones de respuesta y resumen están en `src/analizar_filtros_matematicos.py`; las pruebas verifican coeficientes y algunos resúmenes construidos con valores sintéticos. | La auditoría completa de sensibilidad necesita el recibo y archivos privados del análisis. La demo compara ramas, pero no reconstruye la sensibilidad ni sus resultados para la cohorte. |
| Evaluación Borg/RPE | No hay módulo, esquema de entrada, columnas de puntuación, alineación temporal ni pruebas de Borg/RPE en el árbol remoto revisado. | No se puede evaluar asociación o rendimiento frente a Borg. Se requieren puntuaciones autorizadas, escala identificada, vínculo temporal con cada serie y protocolo de análisis aprobado. |
| Entrenamiento e inferencia del modelo | `src/entrenar_proxy_temporal.py`, evaluadores y `src/serializar_modelo_proxy.py` contienen código para el proxy de tercios temporales; el serializador carga etiquetas, selección anidada y trazas locales. | Los scripts de cohorte exigen archivos privados y fijan el alcance a 15 grupos/414 segmentos. No se distribuye checkpoint ni hay un comando de inferencia con un modelo público verificado. La etiqueta es posición temporal, no fatiga. No se publica un modelo hasta verificar permisos, artefacto, preprocesamiento y evaluación de uso previsto. |
| Pruebas y dependencias | `requirements.txt` fija NumPy, pandas y Pillow; la suite actual tiene 43 pruebas `unittest` y no carga la cohorte privada completa. | La suite comprueba contratos y agregaciones sintéticas, no reproduce resultados F31 ni sustituye validación fisiológica. El árbol remoto no contiene un flujo de CI visible. El smoke agrupado real cubre un registro individual, no la cohorte. |
| README y comandos | `README.md` documenta instalación, suite, ayuda del benchmark, demo, CSV individual y la interfaz CSV de cohorte; `datasets/README.md` restringe el uso de los datos. | Falta un conjunto completo de entradas privadas permitido para reproducir los resultados de cohorte. Antes de ejecutar los módulos privados se requieren datos autorizados, recibos coincidentes, particiones y destino con controles de acceso adecuados. |

## Riesgo de divulgación observado

La búsqueda de archivos del árbol remoto no encontró señales crudas, recibos, filas de participantes ni checkpoints. Las pruebas sintéticas usan los IDs explícitos `synthetic_subject_01` etc. El README menciona una ruta genérica de `%LOCALAPPDATA%`, no una ruta de usuario. Aun así, las salidas de `src/analizar_umbral_senal_emg.py` incluyen identificadores de participante/registro y tiempos de eventos, y los recibos privados de entrenamiento/serialización guardan rutas o hashes locales; no deben publicarse. `src/release_paths.py` solo bloquea entradas/salidas dentro del checkout: no comprueba permisos del directorio, sincronización en nube ni exposición por carpeta compartida. El informe y heatmap publicados contienen métricas agregadas por pierna/serie, no filas individuales; siguen sujetos a la revisión de divulgación institucional aplicable.

## Prerrequisitos para cerrar las brechas

- Para repetir resultados F31: acceso autorizado al inventario original, manifiesto pseudonimizado, señal/recibo de filtros y particiones congeladas, ejecutados en almacenamiento privado fuera del repositorio.
- Para validar eventos mecánicos o fatiga: modalidad y calibración del canal, evento/tarea y fase documentados, medición independiente sincronizada (por ejemplo, fuerza o anotación autorizada) y un criterio preespecificado. Borg requiere además la escala exacta y sus marcas de tiempo alineadas.
- Para liberar inferencia: artefacto y licencia autorizados, contrato de preprocesamiento, prueba de carga del modelo y evaluación en el uso previsto. El paquete actual no incluye esos pesos.

## Alcance científico

La variable objetivo del análisis F31 es el tercio temporal temprano, medio o tardío de cada grabación. Las métricas por pierna describen esa clasificación temporal y no demuestran fatiga fisiológica ni asimetría neuromuscular. No hay etiquetas independientes de fatiga, fuerza/torque sincronizada, sitio anatómico local confirmado ni calibración física de amplitud; no se define umbral, inicio o pico de fatiga. Esta entrega F31 se limita al análisis local de los 15 grupos y no añade validación externa. Las etiquetas del CLI de referencia también son solo tercios de posición temporal.

No se añadió una licencia porque los derechos para otorgarla no se verificaron. La reproducción completa requiere acceso autorizado a las grabaciones locales.


## F44 — Reensamblado funcional y verificación actual (2026-10-19)

- Se añadió `read_cohort_signal_fragments` y la opción `--cohort-fragments`. Los archivos agrupados se validan individualmente, se fusionan por `(subject_id, side, series)` usando sus timestamps originales, se ordenan cronológicamente y se rechazan solapamientos o muestras duplicadas; los huecos se conservan para la segmentación existente.
- La suite local pasó 36/36 pruebas; `compileall` y `--help` pasaron. Las tres pruebas nuevas cubren reordenación, preservación de huecos, rechazo de solapamientos y equivalencia del CLI frente a un archivo completo.
- En almacenamiento restringido, una grabación formal R1 ya autorizada se procesó completa y luego como cuatro CSV contiguos. Las seis tablas de señal (`segments`, `mask`, `features`, `fft`, `psd`, `stft`) coincidieron byte por byte. La ejecución completa generó cinco ramas de filtro; la salida se mantuvo fuera del repositorio.
- Este resultado verifica el reensamblado fuera de línea para una grabación. No demuestra que se haya procesado la cohorte completa, que haya una inferencia Borg funcional, que exista una predicción temporal/Gantt verificable ni que se detecte fatiga fisiológica. No se incluye ningún CSV, señal, salida individual, checkpoint ni hash privado en este paquete.

## F44-04 — Correcciones de seguridad enfocadas (2026-10-20)

La auditoría Codex Security de la revisión `009ba01016fa293b8e3715379f1ec35c6f9812a9` reportó dos hallazgos locales: CWE-22 en rutas derivadas de manifiestos/etiquetas y CWE-1236 en el campo `subject_id` exportado a CSV. Se añadieron resolución de rutas contenidas en la raíz configurada, validación de componentes de nombre de archivo y neutralización de prefijos de fórmula únicamente en el CSV exportado. Las claves originales permanecen intactas en memoria. Una prueba con junction en Windows confirma el rechazo de una ruta que apunta fuera de su raíz; otra recorre las seis tablas agrupadas y confirma que sus identificadores de fórmula se exportan neutralizados.

La revisión independiente post-cambio confirmó que ambas rutas de ataque estáticas están cerradas. El hash final vuelve a resolver el camino bajo la raíz. Permanece una limitación de carrera entre resolver y abrir si otro proceso puede modificar concurrentemente el árbol de entrada; el CLI de rutas no hace esa operación atómica. Las 43 pruebas locales, `compileall` y `pip check` pasaron. Esto verifica de forma dirigida los dos hallazgos; no convierte la cobertura parcial de la auditoría original en una revisión integral del repositorio. Tampoco agrega validación de un modelo Borg, fatiga fisiológica, inferencia en tiempo real, dashboard o Gantt. Las funciones de procesamiento existentes siguen siendo el CLI por lotes documentado arriba.
