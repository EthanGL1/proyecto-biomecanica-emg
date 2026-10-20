# Auditoría del paquete público EMG — estado verificado al 20 de octubre de 2026

**Estado de liberación:** intermedio, seguro para datos y todavía no final científico. El informe histórico previo se conserva en el historial de Git; esta revisión describe el árbol actual de `main` y reemplaza sus conteos de pruebas y manifiesto que habían quedado desactualizados.

## Alcance y privacidad

El repositorio público contiene código de procesamiento, pruebas, instrucciones, referencias y agregados/documentos de liberación permitidos. La inspección del árbol remoto no encontró señales EMG, `participants.csv`, salidas por sujeto, predicciones privadas, etiquetas/fenotipos individuales, datos Cerqueira ni checkpoints restringidos. El último commit auditado antes de esta actualización documental fue `c6f484a1cdc8c98c47c84d7aa54058a556389831`; el autor del commit usa el correo GitHub no-reply. El manifiesto SHA-256 enumera 39 archivos de carga útil (excluye el propio manifiesto) y sus 39 hashes coinciden con los blobs rastreados.

## Comprobaciones reproducibles

- `python -m compileall -q src`: completó sin errores.
- `python -m unittest discover -s tests -v`: **53/53 pruebas aprobadas** en siete módulos.
- `src/reproducir_emg_referencia.py --demo --dashboard`: generó las exportaciones y el visor esperados en modo determinista con señales sintéticas multisinusoidales. Esta comprobación valida ejecución del software; no representa EMG fisiológica, entrenamiento de fatiga ni reproducción de la cohorte.
- Un clon shallow limpio confirmó que el árbol, los hashes del manifiesto y la suite coinciden con el remoto.

## Capacidades presentes y límites

El paquete contiene implementaciones reales de filtros Butterworth y FIR Hamming, manejo de huecos y controles de señal, FFT/PSD/STFT, RMS/MDF/MNF, segmentación exploratoria de actividad, panel/Gantt, lectura y exportación de registros CSV agrupados R1–R5, y código para el benchmark proxy temporal. La modalidad RAW/RECT/ENV, la calibración física, la cadena analógica, la calidad fisiológica y el sitio de adquisición de las grabaciones locales no se verifican mediante la demostración sintética.

El flujo CSV agrupado genera resultados por señal; no reproduce por sí solo las comparaciones de toda la cohorte local de 138 grabaciones. El benchmark de código abierto etiqueta tercios de tiempo. El árbol no ofrece un checkpoint Borg/Cerqueira verificado ni un flujo autónomo de inferencia del modelo. Por tanto, no proporciona probabilidades validadas de fatiga.

El documento PDF/DOCX público es el informe histórico F31, no el manuscrito LaTeX F49 actual. El manuscrito F49 sigue sin PDF compilado e inspeccionado; la compilación no se logró en el entorno TeX autorizado disponible. La validación externa fisiológica tampoco está demostrada: SWIFTIES no cumple el contrato congelado del modelo Borg y no es una validación negativa.

## Criterio de publicación

Esta actualización solo corrige documentación de estado y auditoría. El paquete debe seguir marcado **intermedio** hasta que exista evidencia autorizada de inferencia de modelo compatible, se complete el manuscrito PDF y sus revisiones visuales, y se integren los componentes reproducibles que todavía faltan. No se publican señales privadas, filas individuales, checkpoints restringidos ni afirmaciones de fatiga fisiológica no sustentadas.
