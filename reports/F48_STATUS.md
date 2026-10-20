# Estado del proyecto y del paquete público — corte F49.2 (20 de octubre de 2026)

Este archivo mantiene su nombre histórico `F48_STATUS.md` para no romper referencias. El contenido resume el estado más reciente comprobado y deja claro qué partes siguen pendientes. El paquete de GitHub es una actualización intermedia, no la entrega científica final.

## Estado de cada componente

| Componente | Evidencia actual | Estado y límite |
|---|---|---|
| EMG local de cuádriceps | Inventario de 140 CSV; 138 grabaciones formales R1–R5 y dos R0/R0_1 de control de calidad; 15 grupos derivados de las señales; comparaciones pareadas por pierna y serie con sus denominadores. | Análisis descriptivo ejecutado. La modalidad oscilatoria/rectificada/envolvente y la calibración física no están confirmadas; no hay fuerza, torque ni etiquetas independientes de fatiga funcional. Los cambios espectrales son condicionales y no establecen fatiga. |
| Señal y filtros | Comparación de señal guardada, filtros Butterworth HP20/HP40/BP20–400 y FIR Hamming; FFT, PSD, RMS, MDF, MNF, STFT, máscaras y candidatos de activación documentados en F39/F49. | Implementado y respaldado por fuentes locales. Los candidatos no son repeticiones mecánicas verificadas. Los filtros no se presentan como una referencia limpia ni como ganador fisiológico. |
| Modelo Borg de hombro | Benchmark agrupado OOF para Borg RPE≥17, con las familias y el selector comparados sobre las mismas particiones. | Predice esfuerzo percibido elevado en el protocolo de hombro. La carga/inferencia del checkpoint final no se ha verificado por el bloqueo de la plataforma autorizada; no se reportan predicciones del checkpoint ni fatiga de cuádriceps. |
| Cerqueira de miembro superior | 13 participantes; 146 ensayos y 44.886 ventanas en la cohorte analítica primaria; LOSO agrupado, inferencia interna y paridad entre ejecución completa y fragmentos verificadas. | Objetivo de auto-reporte ordinal de fatiga del miembro superior. El control temporal supera al predictor EMG; no se selecciona un modelo EMG desplegable ni se transfiere al cuádriceps. No es validación entre datasets. |
| Validación externa | SWIFTIES tiene incompatibilidades con la entrada congelada del modelo Borg; no se forzó conversión. | Validación externa fisiológica no demostrada. La incompatibilidad no se interpreta como fallo del modelo. |
| Manuscrito LaTeX | F49.2 incorpora el estudio local, Borg y Cerqueira como ramas separadas, 44 referencias y correcciones de métodos/procedencia. La validación estructural está disponible. | No se ha compilado ni inspeccionado un PDF. El objetivo de extensión y la paginación final no están confirmados. |
| Paquete público GitHub | Código, pruebas, documentación y manifiesto SHA-256; última auditoría de remoto y suite de 53/53 pruebas aprobadas. | Seguro e intermedio. No contiene señales ni checkpoints privados; todavía no contiene un sistema de inferencia Borg/Cerqueira validado ni el manuscrito LaTeX final. |

## Cierre

Los resultados locales y ordinales permanecen separados por tarea y población. Un F1 alto de un objetivo temporal o de esfuerzo percibido no demuestra detección de fatiga fisiológica. Los documentos históricos F31/F48 y sus resultados anteriores permanecen en el historial de Git; este archivo sustituye sus conteos antiguos como resumen de estado actual.
