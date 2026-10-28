# Guion de presentación científica — Grupo 4

**Idioma:** español

**Duración asignada:** 10:00; los tiempos por diapositiva suman 600 segundos. Ensayar en voz alta con cronómetro y mantener las limitaciones aunque haya que ajustar el ritmo.

**Presentación:** diseño Canva existente, “Defensa sEMG cuádriceps”, 10 diapositivas.

**Demostración:** el lanzador F59 procesa descriptivamente un CSV formal R1–R5 o un ZIP con CSV compatibles. Valida los miembros del archivo y mantiene las salidas en una carpeta privada. No carga un modelo predictivo, no verifica repeticiones mecánicas ni estima fatiga fisiológica. No hay CSV individuales autorizados de los dos pacientes para mostrar resultados por paciente.

## Diapositiva 1 — EMG de cuádriceps durante extensión unilateral (00:30)

**Apoyo visual:** título, integrantes y pregunta del estudio.

**Guion hablado:**

Buenos días. Somos el Grupo 4. Estudiamos grabaciones de electromiografía de superficie asociadas con extensión unilateral de rodilla. Nos enfocamos en cómo se preparó y analizó la señal guardada, y en qué cambios aparecen entre series y entre lados. Distinguimos el protocolo que propone la guía de lo que realmente se puede comprobar en los archivos. El resultado es una caracterización de señales; no es un diagnóstico validado de fatiga muscular.

## Diapositiva 2 — Objetivo y protocolo previsto (00:40)

**Apoyo visual:** ejercicio de la guía y músculo objetivo previsto.

**Guion hablado:**

La guía plantea extensión unilateral de rodilla para estudiar el cuádriceps, con el recto femoral como músculo objetivo y una carga prevista del setenta por ciento de una repetición máxima. Estos son elementos del protocolo previsto, no mediciones confirmadas en cada archivo. El objetivo ejecutable fue describir las señales disponibles y evaluar tendencias electromiográficas con reglas reproducibles. Los CSV no confirman la ubicación del electrodo, la carga realmente usada, la fuerza, las pausas, la cadencia ni el momento de cada repetición. Esa diferencia determina hasta dónde podemos interpretar los resultados.

## Diapositiva 3 — Qué contienen los archivos (00:50)

**Apoyo visual:** quince grupos derivados de las carpetas, archivos formales y preliminares, puntos de muestra y combinaciones faltantes.

**Guion hablado:**

El inventario reúne ciento cuarenta archivos CSV de quince grupos de grabación seudonimizados: ciento treinta y ocho series formales R1 a R5, y dos registros preliminares R0 y R0_1 que se mantienen aparte para control de calidad. En conjunto contienen cuatro millones doscientos cincuenta y seis mil doscientos cincuenta y siete puntos de muestra. Faltan doce combinaciones esperadas de participante, pierna y serie. Los archivos contienen tiempo relativo, valor ADC y una columna llamada Voltage_mV. Hay valores ADC negativos y la columna de voltaje es una transformación aritmética; no se verificó una calibración física. Tampoco hay marcadores sincronizados de fuerza, video o repetición. Por eso un pico no se puede llamar repetición confirmada.

## Diapositiva 4 — Preparación de la señal guardada (00:55)

**Apoyo visual:** rama de señal original y filtros comparados.

**Guion hablado:**

Conservamos los archivos originales y separamos los tramos cuando la diferencia entre timestamps supera uno coma cinco milisegundos; no interpolamos a través de esos huecos. Comparamos un pasaaltos Butterworth de cuarto orden a veinte hertz, otro a cuarenta hertz, un pasabanda Butterworth de veinte a cuatrocientos hertz y un FIR de ciento una tomas con ventana Hamming en la misma banda. El filtrado hacia adelante y atrás tiene fase neta cero en los tramos procesables, aunque sus bordes siguen expuestos a transitorios. No aplicamos notch porque la evaluación predefinida de interferencia de red no alcanzó el criterio de diez decibelios. Sin una señal limpia de referencia no podemos afirmar que un filtro haya recuperado la señal fisiológica verdadera mejor que los demás.

## Diapositiva 5 — Características y significado físico (00:50)

**Apoyo visual:** ecuaciones de RMS, MDF, MNF y representación del espectro.

**Guion hablado:**

Para una ventana centrada, el RMS es la raíz cuadrada del promedio de las muestras al cuadrado. Aquí se reporta en cuentas ADC como indicador de amplitud, no en fuerza ni en milivoltios calibrados. La transformada discreta de Fourier suma cada muestra ponderada por una ventana y por una exponencial compleja. La densidad espectral de Welch promedia periodogramas de segmentos; se expresa en cuentas ADC al cuadrado por hertz. La MDF es la frecuencia que divide en dos la potencia integrada, y la MNF es el promedio de frecuencia ponderado por PSD. En una contracción sostenida y controlada, MDF o MNF pueden descender; RMS puede subir, mantenerse o bajar según reclutamiento, esfuerzo y movimiento. Ninguna dirección aislada es un umbral universal de fatiga.

## Diapositiva 6 — Actividad y validez de las métricas espectrales (00:55)

**Apoyo visual:** separación entre reposo y actividad, y comparación de duraciones.

**Guion hablado:**

El análisis fisiológico F39 distingue la ocupación activa de una ventana del solapamiento temporal entre ventanas consecutivas: son parámetros distintos. Se conservaron contracciones débiles si tenían duración continua y calidad suficiente; el reposo no se interpretó con MDF o MNF y no se unieron tramos separados por huecos. A una ocupación del cincuenta por ciento, la cobertura espectral mediana fue veintisiete coma seis por ciento con ventanas de cero coma veinticinco segundos, diecinueve coma uno por ciento con medio segundo y dos coma siete por ciento con un segundo. Al aumentar la duración se perdieron más episodios débiles o tardíos; con un segundo hubo muy pocos datos para estimar dispersión dentro de ventana. El lanzador F57 es otra herramienta: usa Otsu sobre ventanas RMS de cincuenta milisegundos para proponer actividad. Otsu segmenta actividad; no detecta fatiga.

## Diapositiva 7 — Progresión R1 a R5 en la misma pierna (01:15)

**Apoyo visual:** estimaciones con bootstrap por participante y denominadores por métrica.

**Guion hablado:**

Encontramos veintisiete pares R1 frente a R5 del mismo participante y la misma pierna. Los denominadores cambian según la métrica: RMS activo pudo compararse en veintiséis pares de quince participantes; MDF y MNF, en veinte pares de doce participantes. El cambio relativo estimado de RMS fue más ocho coma cinco siete uno por ciento, con intervalo de confianza del noventa y cinco por ciento desde menos cero coma seis ocho uno hasta más dieciocho coma dos seis cuatro. MDF cambió menos cero coma cinco tres cinco hertz, con intervalo desde menos tres coma ocho siete siete hasta más dos coma seis seis tres. MNF cambió menos tres coma tres cero seis hertz, con intervalo desde menos seis coma ocho siete ocho hasta menos cero coma dos siete siete. Solo el intervalo principal de MNF excluye cero, y el resultado depende de la duración y del estimador espectral. Sin fuerza, pausas o intensidad comparables, esto es una tendencia del canal, no fatiga confirmada.

## Diapositiva 8 — Comparación entre piernas (01:15)

**Apoyo visual:** pares izquierda-derecha y sus intervalos.

**Guion hablado:**

Había hasta sesenta y tres pares izquierda-derecha por serie formal. En las comparaciones evaluables, RMS estuvo disponible para sesenta pares de catorce participantes y MDF y MNF para cuarenta y nueve pares de doce participantes. El contraste bilateral de RMS fue menos cero coma tres cuatro dos por ciento, con intervalo del noventa y cinco por ciento entre menos nueve coma ocho seis cinco y más ocho coma ocho tres siete. MDF fue más cero coma nueve nueve cuatro hertz, intervalo entre menos tres coma nueve seis nueve y más cinco coma seis uno ocho. MNF fue más uno coma uno cero cuatro hertz, intervalo entre menos tres coma siete cinco uno y más cinco coma seis siete cinco. Los tres intervalos incluyen cero: no se detectó una diferencia consistente. Eso no demuestra equivalencia fisiológica entre piernas; la lateralidad proviene del nombre del archivo y no se midió la biomecánica.

## Diapositiva 9 — Benchmark de posición temporal (01:00)

**Apoyo visual:** validación LOSO agrupada y definición exacta del objetivo.

**Guion hablado:**

El benchmark local previamente ejecutado clasifica si una ventana pertenece al primer, segundo o último tercio temporal de la grabación. Usa validación leave-one-subject-out agrupada por participante y reportó Macro-F1 de cero coma siete uno seis, con desviación de cero coma uno cuatro seis. Se compararon mayoría, centroide, k vecinos, ridge, regresión logística, LDA, SVM lineal, random forest, boosting por árboles cortos y modelos RMS/MDF con LDA o logística. El vector principal tuvo quince características: media ADC, desviación estándar poblacional, RMS, media absoluta, mediana, desviación absoluta mediana, percentiles 5/25/75/95, pico a pico, longitud de línea media, cruces de mediana, asimetría y curtosis excedente. Es un resultado de clasificación temporal, no de repeticiones, pérdida de fuerza, comienzo de fatiga ni nivel fisiológico. El lanzador F59 no carga ese benchmark ni ningún otro clasificador. Los modelos Borg y Cerqueira corresponden a objetivos de esfuerzo percibido o auto-reporte en protocolos de miembro superior; no validan fatiga del cuádriceps. Para seleccionar un detector de repeticiones se requieren anotaciones mecánicas sincronizadas. Para validar fatiga se requiere una referencia independiente adecuada.

## Diapositiva 10 — Conclusiones y demostración local (01:50)

**Apoyo visual:** resultados principales, límites y lanzador en la carpeta del proyecto.

**Guion hablado:**

La fatiga puede modificar la actividad de las unidades motoras y el espectro EMG. En una contracción sostenida y controlada, MDF o MNF pueden descender cuando cambia la conducción de las fibras. Pero en una tarea dinámica también cambian el ángulo, el reclutamiento, el esfuerzo y la posición del electrodo. Por eso RMS no sigue una sola dirección y una tendencia espectral no demuestra pérdida de fuerza.

En R1 a R5 observamos un aumento medio de RMS activo y descensos medios de MDF y MNF. Solo el intervalo principal de MNF excluyó cero, con sensibilidad a la duración espectral. Los intervalos bilaterales incluyeron cero. No verificamos la modalidad física de la señal, calibración de voltaje, sitio del electrodo, carga real, pausas, cadencia ni marcadores mecánicos. El estudio describe tendencias del canal guardado; no confirma fatiga del recto femoral ni su inicio.

Para la demostración, hacemos doble clic en `INICIAR_INFERENCIA.bat`. En el primer uso instala el entorno y ejecuta las comprobaciones. Elegimos un CSV formal R1 a R5 o un ZIP con los CSV compatibles, y una carpeta privada de salida. El programa genera un manifiesto por grabación, eventos candidatos, un índice y tableros. Esta ejecución de un clic es descriptiva: no carga un modelo, no confirma repeticiones ni predice fatiga.

No se suministraron CSV individuales autorizados de los dos pacientes; por eso no mostramos una inferencia ni resultados de esas personas. La prueba real registrada procesó una grabación de desarrollo y no es validación externa. Para validar repeticiones hacen falta marcas mecánicas sincronizadas; para validar fatiga, una referencia independiente y compatible. Muchas gracias.
