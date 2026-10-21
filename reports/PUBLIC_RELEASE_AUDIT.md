# Auditoría del paquete público EMG — corte actual F51

## Verificación actual F51

El corte F51 parte del main público F50.9 (a5966d46d3b683b024e738be68440c545a52f4a2). El paquete mantiene visibilidad pública y separa el código de las señales y resultados individuales.

- En un runtime Python 3.12.14, 86/86 pruebas, compileall, ayuda CLI y demo sintética terminaron correctamente.
- Un smoke privado de una copia de R1 real verificó señal completa, un fragmento, reensamblado de tres fragmentos, un hueco artificial y rechazo de duplicado. Completa y reensamblado produjeron las mismas seis tablas, eventos y tablero. El resumen público incluye conteos agregados, no la señal, los archivos resultantes ni rutas personales.
- El control de hueco omitió intencionalmente 40 muestras y preservó un salto de 41 ms en dos tramos. No representa una anomalía observada en la grabación original.
- La prueba no equivale a evaluación de los 138 registros, no carga un modelo y no confirma fisiología o fatiga.
- No se añadieron señales, salidas por participante, checkpoints, credenciales ni el manuscrito completo. La figura individual Cerqueira F51 sigue privada.

Conclusión: el paquete demuestra ejecución del procesador EMG descriptivo para un archivo real y consistencia de reensamblado. La entrega pública sigue intermedia: el PDF actual no se exportó ni inspeccionó y Codex Security no produjo un informe formal.

## Registro histórico F50.9 conservado

### Verificación F50.9

Este corte sigue al commit F50.8 `15bd1fafb02b2ca532b0d29ee7bad0bb47141b79` en la rama pública `main`. El repositorio permanece público. La actualización agrega un estado de verificación y corrige la descripción del PDF y del estado del escaneo; no agrega código, señales, pesos de modelo ni filas por sujeto.

- El snapshot F50.8 tenía 53 archivos: el manifiesto enumeraba 52 cargas útiles y se excluía a sí mismo. F50.9 agrega un archivo de estado; el árbol esperado queda en 54 archivos y el manifiesto enumera 53. Se actualizaron los hashes de README y de esta auditoría y se añadió el hash del nuevo estado.
- La suite local verificable dio 86/86 pruebas aprobadas; `compileall` devolvió 0; las interfaces de ayuda devolvieron 0. La demo sintética produjo las diez exportaciones previstas y dejó explícito `fatigue_target: none` y abstención supervisada.
- El checkpoint Cerqueira full-fit privado cargó y ejecutó cinco ventanas sintéticas con probabilidades válidas. La prueba es de serialización/preprocesamiento/CLI; no es evaluación etiquetada ni generalización. El checkpoint y sus salidas no se publican.
- El manuscrito F50.9 compiló en el editor integrado, pero no existe PDF exportado, páginas contadas ni revisión visual. El intento local de exportación fue bloqueado por Device Guard (error 4551).
- El escaneo formal Codex Security no se cerró: Windows Application Control impidió cargar una extensión nativa del ayudante; no hay informe sellado ni afirmación de cero hallazgos.

**Conclusión actual:** la publicación es intermedia y privada en datos. La demo de software funciona, pero no sustituye la inferencia del modelo sobre datos etiquetados ni valida fatiga del cuádriceps. El informe se mantiene abierto hasta que se exporte e inspeccione el PDF y se complete la revisión formal de seguridad en un entorno permitido.

## Registro histórico F50.8 conservado

# Auditoría del paquete público EMG — estado verificado al 20 de octubre de 2026

**Estado de liberación:** intermedio, sin señales privadas en el paquete y todavía no final científico. Este checkout parte de `origin/main` `da04af2cbc653a6849726b4b9119af0f3474cd41`. La actualización de código y pruebas se publicó en `1937a53f5bd780b9592ae216ed2b3a0f5888743b`; después del envío, `git ls-remote` coincidió con ese SHA. Esta auditoría se actualiza con los resultados de esa publicación. Los SHAs `a9a…` y `c6f…` de párrafos históricos identifican snapshots anteriores.

## Alcance y privacidad

La auditoría remota anterior no encontró señales EMG, `participants.csv`, salidas por sujeto, predicciones privadas, etiquetas/fenotipos individuales, datos Cerqueira ni checkpoints restringidos. Ese snapshot histórico tenía 39 archivos de carga útil. El paquete de código publicado incluye 47 archivos de carga útil; en el árbol versionado de esta auditoría, los 47 hashes del manifiesto coinciden con los blobs de sus respectivos archivos. La inspección del cambio publicado confirma que solo agrega código, pruebas sintéticas y documentación; no se incluyen datos ni checkpoints.

## Comprobaciones reproducibles

- Python 3.12.14 del runtime de trabajo: `python -m compileall -q src tests` completó sin errores.
- `python -m unittest discover -s tests`: **69/69 pruebas aprobadas** en ocho módulos, incluida la limpieza de salidas opcionales obsoletas en llamada directa y mediante `--overwrite`, el pipeline Cerqueira sintético completo, protección de destinos y neutralización de fórmulas CSV.
- `src/reproducir_emg_referencia.py --demo --dashboard`: generó las exportaciones y el visor esperados en modo determinista con señales sintéticas multisinusoidales. Esta comprobación valida ejecución del software; no representa EMG fisiológica, entrenamiento de fatiga ni reproducción de la cohorte.
- Los resúmenes agregados de cohorte se comprobaron con fixtures sintéticos: reducción de ventanas por grabación, exclusión de pares incompletos, rechazo de R0, conservación de RMS con MDF/MNF ausentes y ausencia de pseudónimos en las tres tablas agregadas. No se procesó ni recalculó la cohorte local en esta extensión.
- La revisión de código detectó productos opcionales obsoletos al reutilizar la carpeta; se corrigió la escritura para retirar las tablas de cohorte y el tablero anteriores cuando la nueva ejecución no los genera. La prueba de CLI reproduce la secuencia con datos sintéticos y verifica su eliminación. La revisión formal de Codex Security quedó incompleta: Windows Application Control bloqueó el módulo nativo requerido por la fase de descubrimiento y no se intentó eludir esa política. Por ello no se presenta un resultado formal de “sin hallazgos”.

## Capacidades presentes y límites

El paquete contiene implementaciones reales de filtros Butterworth y FIR Hamming, manejo de huecos y controles de señal, FFT/PSD/STFT, RMS/MDF/MNF, segmentación exploratoria de actividad, panel/Gantt, lectura y exportación de registros CSV agrupados R1–R5, y código para el benchmark proxy temporal. La modalidad RAW/RECT/ENV, la calibración física, la cadena analógica, la calidad fisiológica y el sitio de adquisición de las grabaciones locales no se verifican mediante la demostración sintética.

El flujo CSV agrupado genera resultados por señal y tres resúmenes agregados sin identificadores directos (`cohort_by_side_series.csv`, `cohort_r1_r5_paired.csv`, `cohort_bilateral_paired.csv`), primero reduciendo ventanas a medianas por grabación y luego resumiendo por participante; los contrastes emparejados solo usan pares completos. La agregación no garantiza anonimización y los agregados deben permanecer restringidos cuando el denominador sea pequeño. La escritura ahora elimina esos tres productos al reutilizar la carpeta para una entrada no agrupada, y retira un `dashboard.html` previo si el nuevo resultado no solicita tablero. Es una utilidad general para entradas autorizadas y no reproduce por sí sola las comparaciones F38/F39 de la cohorte local de 138 grabaciones: carece de la máscara de actividad y de varios criterios locales de elegibilidad. Los archivos por señal conservan seudónimos. El benchmark de código abierto etiqueta tercios de tiempo. El árbol no ofrece un checkpoint Borg/Cerqueira verificado ni un flujo autónomo de inferencia del modelo. Por tanto, no proporciona probabilidades validadas de fatiga.

El documento PDF/DOCX público es el informe histórico F31, no el manuscrito LaTeX F49 actual. El manuscrito F49 sigue sin PDF compilado e inspeccionado; la compilación no se logró en el entorno TeX autorizado disponible. La validación externa fisiológica tampoco está demostrada: SWIFTIES no cumple el contrato congelado del modelo Borg y no es una validación negativa.

## Addendum F50 — pipeline Cerqueira de miembro superior

El estado anterior de esta auditoría describía el paquete previo a la incorporación del pipeline Cerqueira. La actualización pública ahora incluye cuatro módulos de fuente y pruebas sintéticas para auditoría de alineación, extracción de características, evaluación anidada LOSO e inferencia de pliegue. La documentación de ejecución y privacidad está en `src/cerqueira/README.md`. No se publicaron datos fuente, modelos, predicciones OOF ni resultados granulares. El pipeline permite repetir el análisis desde archivos externos autorizados; las pruebas automáticas no descargan ni procesan el corpus oficial.

La revisión estática independiente identificó colisiones y riesgo de sobrescritura en rutas de salida. Se corrigieron: cada par de salidas debe resolver a archivos diferentes y nuevos; auditoría y extracción también rechazan rutas hacia sus entradas; inferencia rechaza rutas hacia entradas, checkpoint y manifiesto, y prepara ambos directorios antes de escribir. Todos los comandos rechazan productos dentro del checkout público. Las pruebas de regresión verifican estos contratos. La ubicación sigue dependiendo del usuario: las salidas por sujeto/ensayo/ventana deben estar fuera del checkout y de carpetas sincronizadas o compartidas.

Una revisión focalizada reprodujo que un nombre de miembro ZIP con prefijo de fórmula podía llegar a la hoja CSV del auditor. El escritor ahora neutraliza esos prefijos en el CSV y conserva los nombres exactos en JSON; el caso de extremo a extremo y variantes de prefijos pasan en pruebas sintéticas. El escaneo formal Codex Security no se completó porque Windows Application Control bloqueó su componente nativo durante el registro de descubrimientos. Además, el scan capturó el snapshot previo a la corrección (`codex-security-snapshot/v1:sha256:e52139181e240c4a2ff56ea66de429a4e36e1b10718b331534819544d0c1be65`), por lo que el código final no cuenta con revisión formal de ese scan. La corrección manual y las pruebas no sustituyen ni se presentan como un escaneo sellado.

El pipeline de inferencia carga solo artefactos de pliegues LOSO verificados por hash. No se publica un modelo final apto para despliegue ni se recomienda usarlo, dado que el comparador temporal supera al predictor EMG en la evaluación disponible. La revisión de código y las pruebas no constituyen una auditoría formal de Codex Security.

## Criterio de publicación

Esta actualización agrega resúmenes descriptivos de cohorte al procesador público; no cambia filtros, extracción de características, datos históricos ni entrenamiento. El paquete debe seguir marcado **intermedio** hasta que exista evidencia autorizada de inferencia de modelo compatible, se complete el manuscrito PDF y sus revisiones visuales, y se integren los componentes reproducibles que todavía faltan. No se publican señales privadas, filas individuales, checkpoints restringidos ni afirmaciones de fatiga fisiológica no sustentadas.
