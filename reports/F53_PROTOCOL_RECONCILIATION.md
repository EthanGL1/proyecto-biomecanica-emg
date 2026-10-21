# F53 — Reconciliación del protocolo previsto y los datos locales

Fecha: 21 de octubre de 2026. Estado: corrección de alcance científico; actualización pública intermedia.

## Pregunta del proyecto

La guía docente *Integrative Project — Guidelines* define explícitamente el problema como clasificar el estado de fatiga a partir de sEMG del **rectus femoris (recto femoral)** durante **extensión unilateral de rodilla al 70 % de 1RM**. También define un protocolo bilateral que ejercita una pierna cada vez. Por ello, era demasiado amplia la frase anterior que decía que el proyecto no trataba el impacto de un grupo muscular específico: el objetivo sí es específico. El recto femoral es uno de los músculos del cuádriceps, por lo que el nombre breve “estudio de cuádriceps” es admisible si se conserva esa precisión anatómica.

## Protocolo especificado frente a evidencia archivada

| Elemento | Lo que especifica la guía | Lo que confirma el archivo local revisado | Alcance de la afirmación |
|---|---|---|---|
| Músculo y tarea | Recto femoral bilateral; extensión de rodilla unilateral, una pierna a la vez; objetivo de 70 % de 1RM | El inventario contiene 15 grupos seudónimos y archivos codificados por lateralidad; el CSV no trae un campo de sitio anatómico ni carga | El objetivo es recto femoral/extensión unilateral. El sitio del electrodo y la carga efectivamente usados no se verifican por grabación |
| Electrodos | Ubicación recomendada sobre recto femoral; separación de 20 mm y referencia anatómica, con fotografía de colocación | No se incluyeron fotos de colocación ni metadatos por registro en el archivo auditado | Especificación de protocolo, no comprobación de colocación individual |
| Hardware y modalidad | MyoWare 2.0 de dos canales y salida RAW | Los archivos locales tienen `Relative_Time_s`, `ADC_Value` y `Voltage_mV`; no incluyen configuración del sensor ni modo del pin de salida | La salida RAW fue prescrita, pero la modalidad guardada no se confirma por el nombre de columnas ni por su aspecto |
| Frecuencia de adquisición | Al menos 1000 Hz por canal, controlada por temporizador | La mediana del intervalo entre timestamps es cercana a 1 ms en el inventario | Compatible con una tasa nominal cercana a 1 kHz; no prueba el reloj físico, jitter efectivo ni ancho de banda analógico |
| Codificación/calibración | ADC de 12 bits o superior previsto; la guía no sustituye la calibración del equipo | `Voltage_mV` coincide aritméticamente con `round(ADC_Value × 3300/4095, 2)`; `ADC_Value` incluye negativos | La fórmula de exportación no valida voltios físicos, referencia, ganancia, offset, linealidad ni límites del ADC |
| Ejercicio y progresión | Repeticiones a 70 % de 1RM hasta fallo técnico; 2 s concéntricos/2 s excéntricos; 10 min entre piernas; orden aleatorizado; Borg CR-10 cada cinco repeticiones | Los timestamps y amplitudes no identifican carga, fase mecánica, cadencia, descanso, índice de repetición, Borg ni razón de terminación | La guía define la intención del protocolo. La descripción del usuario de llegar al fallo técnico es contexto reportado, no un evento sincronizado en el archivo |
| Referencias funcionales | La guía contempla MVC pre/post, video y metadatos para apoyar la caracterización | El ZIP de grabaciones auditado no contiene esos archivos independientes ni marcadores sincronizados | No hay referencia local independiente que confirme pérdida de fuerza o una etiqueta fisiológica de fatiga |
| Tamaño del análisis local | Cohorte del curso compartida por los grupos | 140 CSV agrupados en 15 códigos de grabación; 138 formales R1–R5 y dos R0/R0_1 para control de calidad | Se analiza la cohorte de 15 grupos disponible. R0/R0_1 no se mezclan con las cinco series formales |

La asignación `iz`/`dr` a izquierda/derecha y R1–R5 a las cinco series proviene de la información suministrada por el usuario. No se usaron datos demográficos ni de participantes como predictores o etiquetas. El código público acepta registros autorizados y puede resumir entradas agrupadas; la reproducción de los resultados confidenciales de los 15 participantes requiere acceso local autorizado a las señales y sus recibos, no incluidos en GitHub.

## Formulación corregida de los resultados y límites

El proyecto estudia un objetivo anatómico y funcional definido —sEMG de recto femoral durante extensión unilateral de rodilla—, pero los resultados disponibles caracterizan la señal guardada y sus cambios entre tiempo/series. Debido a que el archivo no permite comprobar el sitio por persona, la salida RAW efectiva, calibración, carga y cinemática, y tampoco contiene fuerza/MVC sincronizada o anotaciones mecánicas, no se afirma que una pendiente de MDF/MNF, un evento candidato o una probabilidad proxy sea fatiga fisiológica confirmada del cuádriceps.

En particular:

- “Recto femoral durante extensión unilateral” es el **objetivo especificado por la guía**.
- “Canal EMG local correspondiente al recto femoral” no queda **verificado por archivo para cada registro**.
- La mediana de timestamps apoya una estimación nominal próxima a 1 kHz, pero no la frecuencia efectiva de adquisición completa ni la respuesta del sistema analógico.
- El modelo local de tercios temporales clasifica posición dentro de la grabación; su F1 no mide detección de fatiga.
- El predictor Borg y el corpus Cerqueira tienen objetivos de percepción de esfuerzo/fatiga de otras tareas y segmentos corporales; no se transfieren al resultado fisiológico local.

Esta corrección mejora la especificidad del planteamiento sin ocultar las limitaciones de medición. No se añadieron señales, resultados individuales, identificadores, checkpoints ni información demográfica al repositorio público.
