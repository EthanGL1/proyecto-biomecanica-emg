# Auditoría del paquete público EMG — estado verificado al 20 de octubre de 2026

**Estado de liberación:** intermedio, sin señales privadas en el paquete y todavía no final científico. El informe histórico previo se conserva en el historial de Git. `origin/main` se verificó en `a9a773cb04ade9e8227b4f1dacc4420aede05bca` justo antes de integrar esta actualización; las verificaciones corresponden al árbol probado y el commit remoto se confirma por separado.

## Alcance y privacidad

El repositorio público contiene código de procesamiento, pruebas, instrucciones, referencias y agregados/documentos de liberación permitidos. La inspección del árbol remoto no encontró señales EMG, `participants.csv`, salidas por sujeto, predicciones privadas, etiquetas/fenotipos individuales, datos Cerqueira ni checkpoints restringidos. El último commit auditado antes de esta actualización documental fue `c6f484a1cdc8c98c47c84d7aa54058a556389831`; el autor del commit usa el correo GitHub no-reply. El manifiesto SHA-256 enumera 39 archivos de carga útil (excluye el propio manifiesto) y sus 39 hashes coinciden con los blobs rastreados.

## Comprobaciones reproducibles

- Python 3.12.14 del runtime de trabajo: `python -m compileall -q src tests` completó sin errores.
- `python -m unittest discover -s tests`: **59/59 pruebas aprobadas** en siete módulos, incluida la limpieza de salidas opcionales obsoletas en llamada directa y mediante `--overwrite`, y el rechazo de métricas textuales malformadas.
- `src/reproducir_emg_referencia.py --demo --dashboard`: generó las exportaciones y el visor esperados en modo determinista con señales sintéticas multisinusoidales. Esta comprobación valida ejecución del software; no representa EMG fisiológica, entrenamiento de fatiga ni reproducción de la cohorte.
- Los resúmenes agregados de cohorte se comprobaron con fixtures sintéticos: reducción de ventanas por grabación, exclusión de pares incompletos, rechazo de R0, conservación de RMS con MDF/MNF ausentes y ausencia de pseudónimos en las tres tablas agregadas. No se procesó ni recalculó la cohorte local en esta extensión.
- La revisión de código detectó productos opcionales obsoletos al reutilizar la carpeta; se corrigió la escritura para retirar las tablas de cohorte y el tablero anteriores cuando la nueva ejecución no los genera. La prueba de CLI reproduce la secuencia con datos sintéticos y verifica su eliminación. La revisión formal de Codex Security quedó incompleta: Windows Application Control bloqueó el módulo nativo requerido por la fase de descubrimiento y no se intentó eludir esa política. Por ello no se presenta un resultado formal de “sin hallazgos”.

## Capacidades presentes y límites

El paquete contiene implementaciones reales de filtros Butterworth y FIR Hamming, manejo de huecos y controles de señal, FFT/PSD/STFT, RMS/MDF/MNF, segmentación exploratoria de actividad, panel/Gantt, lectura y exportación de registros CSV agrupados R1–R5, y código para el benchmark proxy temporal. La modalidad RAW/RECT/ENV, la calibración física, la cadena analógica, la calidad fisiológica y el sitio de adquisición de las grabaciones locales no se verifican mediante la demostración sintética.

El flujo CSV agrupado genera resultados por señal y tres resúmenes agregados sin identificadores directos (`cohort_by_side_series.csv`, `cohort_r1_r5_paired.csv`, `cohort_bilateral_paired.csv`), primero reduciendo ventanas a medianas por grabación y luego resumiendo por participante; los contrastes emparejados solo usan pares completos. La agregación no garantiza anonimización y los agregados deben permanecer restringidos cuando el denominador sea pequeño. La escritura ahora elimina esos tres productos al reutilizar la carpeta para una entrada no agrupada, y retira un `dashboard.html` previo si el nuevo resultado no solicita tablero. Es una utilidad general para entradas autorizadas y no reproduce por sí sola las comparaciones F38/F39 de la cohorte local de 138 grabaciones: carece de la máscara de actividad y de varios criterios locales de elegibilidad. Los archivos por señal conservan seudónimos. El benchmark de código abierto etiqueta tercios de tiempo. El árbol no ofrece un checkpoint Borg/Cerqueira verificado ni un flujo autónomo de inferencia del modelo. Por tanto, no proporciona probabilidades validadas de fatiga.

El documento PDF/DOCX público es el informe histórico F31, no el manuscrito LaTeX F49 actual. El manuscrito F49 sigue sin PDF compilado e inspeccionado; la compilación no se logró en el entorno TeX autorizado disponible. La validación externa fisiológica tampoco está demostrada: SWIFTIES no cumple el contrato congelado del modelo Borg y no es una validación negativa.

## Criterio de publicación

Esta actualización agrega resúmenes descriptivos de cohorte al procesador público; no cambia filtros, extracción de características, datos históricos ni entrenamiento. El paquete debe seguir marcado **intermedio** hasta que exista evidencia autorizada de inferencia de modelo compatible, se complete el manuscrito PDF y sus revisiones visuales, y se integren los componentes reproducibles que todavía faltan. No se publican señales privadas, filas individuales, checkpoints restringidos ni afirmaciones de fatiga fisiológica no sustentadas.
