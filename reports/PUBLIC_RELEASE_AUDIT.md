# Auditoría del paquete público EMG — F31

**Fecha:** 19 de octubre de 2026
**Estado:** remoto `main` verificado; esta matriz de brechas es una revisión local sin publicar.

**Referencia remota revisada:** `main` apunta a `24266c044e9ce4c4a4c0493487a5ab7fa6f93def` (`Record public EMG release verification`). La matriz siguiente es una revisión local de brechas del candidato; sus cambios no se han enviado a GitHub.

## Contenido y exclusiones

El paquete contiene 17 módulos Python, dos bibliografías consolidadas, instrucciones, el informe IEEE público F31 en DOCX/PDF, el heatmap y el resumen agregado por pierna/serie. No contiene archivos de participantes, señales del estudio locales o externas, métricas por persona/pliegue/canal, etiquetas, características, particiones, predicciones, recibos privados, modelos serializados ni figuras de caso. `participants.csv` se excluyó del paquete; se utilizó para reconciliación administrativa del proyecto, pero no fue entrada del análisis de señales/características ni fuente de covariables. El CLI de referencia crea datos sintéticos solo cuando se ejecuta.

El informe público conserva resultados agregados locales. Se eliminaron las figuras de R0/R0_1, los puntos de rendimiento por participante y la señal de una grabación individual; también se borraron del DOCX los medios incrustados sin uso. Las 44 referencias consolidadas permanecen sin cambios.

## Controles verificados

- Los 17 módulos compilan; los módulos de lateralidad, filtros y referencia rechazan rutas de entrada o salida dentro del checkout público.
- Las 28 pruebas de `unittest` cubren agregación por persona, heatmap izquierda/derecha, máscara y segmentación con fixtures sintéticas, filtros, RMS/MDF/MNF, FFT/PSD/STFT y validación de entradas. `--help` no lee grabaciones ni entrena.
- `src/reproducir_emg_referencia.py --demo` procesa fixtures de software multisinusoidales R1–R5 (no un simulador fisiológico de EMG) y deja outputs temporales fuera del checkout; no genera resultados ni métricas de la cohorte F31.
- DOCX con 44 referencias, 15 tablas nativas y ocho figuras agregadas o esquemas editables; sin checklist de envío, placeholders ni contenido individual de las tres figuras retiradas.
- PDF exportado desde Word en A4, con 22 páginas renderizadas e inspeccionadas; no se detectaron desbordes ni tablas rotas.
- Las dos bibliografías se cotejaron con las versiones consolidadas y no se modificaron.
- La matriz por lado reporta 14 grupos izquierdos y 15 derechos, con 14 pareados; la diferencia Macro-F1 incluye cero.
- El manifiesto `SHA256SUMS` contiene 33 entradas y sus 33 archivos se verificaron sin discrepancias. La actualización pública se envió a `main` y se comprobó el head remoto tras el push.

## Matriz de brechas de reproducibilidad

| Componente | Código y evidencia disponible | Límite de reproducción pública |
|---|---|---|
| Filtrado y calidad de señal | `src/comparar_filtros_emg.py` separa gaps, diseña/aplica filtros IIR y FIR, y calcula máscaras de bordes; `src/reproducir_emg_referencia.py` comprueba jitter temporal y ejecuta cinco ramas sobre una fixture sintética o un CSV R1–R5 sin columnas adicionales. | Reproduce el algoritmo numérico, no verifica que el canal sea EMG cruda, la calibración, el sitio muscular, los artefactos fisiológicos o la calidad del movimiento. No hay referencia limpia para medir SNR o declarar un filtro ganador. |
| FFT, PSD y STFT | La herramienta de referencia exporta `fft.csv`, `psd.csv` y `stft.csv`; `tests/test_public_emg_pipeline.py` comprueba frecuencia/potencia y las filas generadas. | Reproducible con datos sintéticos o CSV autorizado sin identidad. No reproduce por sí solo el espectro agregado de la cohorte. |
| RMS, MDF y MNF | La misma herramienta calcula estas características sobre ventanas de 1 s con 50 % de solapamiento y banda 20–400 Hz; la prueba sintética comprueba valores esperados. | Unidades ADC, sin calibración física. Las tendencias de cohorte de F31 publicadas son resultados agregados, no se pueden recalcular desde los archivos de este repositorio. |
| Series R1–R5 | El lector público acepta únicamente las etiquetas R1, R2, R3, R4 y R5; la fixture de demo contiene las cinco. | Las etiquetas son nombres de series. No codifican repeticiones, fases de ejercicio ni sucesos mecánicos verificados. No se incluyen grabaciones del estudio. |
| Eventos o umbrales | `src/analizar_umbral_senal_emg.py` implementa ventanas válidas, umbrales robustos y cruces sostenidos candidatos de RMS/MDF. El CLI completo requiere `manifest_pseudonymized.csv`, `receipt_filter_sensitivity_v1.json`, señales pseudonimizadas y archivos filtrados; escribe filas por registro/ventana y figuras individuales en su salida privada. | No se ejecuta desde la demo pública ni con los archivos publicados. Los cruces son descensos exploratorios respecto a ventanas tempranas; no detectan repeticiones, inicio/pico de fatiga, fallo técnico ni eventos validados frente a vídeo, fuerza o anotación independiente. |
| Sensibilidad a filtros | Las funciones de respuesta y resumen están en `src/analizar_filtros_matematicos.py`; las pruebas verifican coeficientes y algunos resúmenes construidos con valores sintéticos. | La auditoría completa de sensibilidad necesita el recibo y archivos privados del análisis. La demo compara ramas, pero no reconstruye la sensibilidad ni sus resultados para la cohorte. |
| Evaluación Borg/RPE | No hay módulo, esquema de entrada, columnas de puntuación, alineación temporal ni pruebas de Borg/RPE en el árbol remoto revisado. | No se puede evaluar asociación o rendimiento frente a Borg. Se requieren puntuaciones autorizadas, escala identificada, vínculo temporal con cada serie y protocolo de análisis aprobado. |
| Entrenamiento e inferencia del modelo | `src/entrenar_proxy_temporal.py`, evaluadores y `src/serializar_modelo_proxy.py` contienen código para el proxy de tercios temporales; el serializador carga etiquetas, selección anidada y trazas locales. | Los scripts de cohorte exigen archivos privados y fijan el alcance a 15 grupos/414 segmentos. No se distribuye checkpoint ni hay un comando de inferencia con un modelo público verificado. La etiqueta es posición temporal, no fatiga. No se publica un modelo hasta verificar permisos, artefacto, preprocesamiento y evaluación de uso previsto. |
| Pruebas y dependencias | `requirements.txt` fija NumPy, pandas y Pillow; las 28 pruebas `unittest` usan fixtures o filas construidas en memoria y no cargan señales del estudio. | La suite comprueba contratos y agregaciones sintéticas, no reproduce resultados F31 ni sustituye validación fisiológica. El árbol remoto no contiene un flujo de CI visible. |
| README y comandos | `README.md` documenta instalación, suite, ayuda del benchmark y demo/CSV de referencia; `datasets/README.md` restringe el uso de los datos. | Falta un conjunto completo de entradas privadas permitido para reproducir los resultados de cohorte. Antes de ejecutar los módulos privados se requieren datos autorizados, recibos coincidentes, particiones y destino con controles de acceso adecuados. |

## Riesgo de divulgación observado

La búsqueda de archivos del árbol remoto no encontró señales crudas, recibos, filas de participantes ni checkpoints. Las pruebas sintéticas usan los IDs explícitos `synthetic_subject_01` etc. El README menciona una ruta genérica de `%LOCALAPPDATA%`, no una ruta de usuario. Aun así, las salidas de `src/analizar_umbral_senal_emg.py` incluyen identificadores de participante/registro y tiempos de eventos, y los recibos privados de entrenamiento/serialización guardan rutas o hashes locales; no deben publicarse. `src/release_paths.py` solo bloquea entradas/salidas dentro del checkout: no comprueba permisos del directorio, sincronización en nube ni exposición por carpeta compartida. El informe y heatmap publicados contienen métricas agregadas por pierna/serie, no filas individuales; siguen sujetos a la revisión de divulgación institucional aplicable.

## Prerrequisitos para cerrar las brechas

- Para repetir resultados F31: acceso autorizado al inventario original, manifiesto pseudonimizado, señal/recibo de filtros y particiones congeladas, ejecutados en almacenamiento privado fuera del repositorio.
- Para validar eventos mecánicos o fatiga: modalidad y calibración del canal, evento/tarea y fase documentados, medición independiente sincronizada (por ejemplo, fuerza o anotación autorizada) y un criterio preespecificado. Borg requiere además la escala exacta y sus marcas de tiempo alineadas.
- Para liberar inferencia: artefacto y licencia autorizados, contrato de preprocesamiento, prueba de carga del modelo y evaluación en el uso previsto. El paquete actual no incluye esos pesos.

## Alcance científico

La variable objetivo del análisis F31 es el tercio temporal temprano, medio o tardío de cada grabación. Las métricas por pierna describen esa clasificación temporal y no demuestran fatiga fisiológica ni asimetría neuromuscular. No hay etiquetas independientes de fatiga, fuerza/torque sincronizada, sitio anatómico local confirmado ni calibración física de amplitud; no se define umbral, inicio o pico de fatiga. Esta entrega F31 se limita al análisis local de los 15 grupos y no añade validación externa. Las etiquetas del CLI de referencia también son solo tercios de posición temporal.

No se añadió una licencia porque los derechos para otorgarla no se verificaron. La reproducción completa requiere acceso autorizado a las grabaciones locales.
