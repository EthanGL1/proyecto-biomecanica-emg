# Análisis de señales EMG y progresión temporal

Este repositorio contiene código, bibliografía e informe público agregado para auditar grabaciones EMG y estudiar rasgos temporales y espectrales. El modelo local clasifica tercios de tiempo del archivo; no es un detector de fatiga fisiológica.

## Alcance y datos

El análisis local usó exclusivamente 140 grabaciones EMG correspondientes a 15 grupos derivados de los códigos de archivo. Los 138 registros R1–R5 participaron en el benchmark de tercios temporales; R0/R0_1 se conservaron para control de calidad. No se leyó ni usó `participants.csv` ni se emplearon peso, altura u otros metadatos de participantes. Las particiones mantuvieron todas las grabaciones de cada persona en un mismo grupo.

No se publican señales, datos por participante, características, etiquetas, particiones, predicciones, resultados por pliegue ni recibos privados. Las mediciones originales y salidas con granularidad individual permanecen en el almacenamiento local restringido. El benchmark completo requiere acceso autorizado a los datos. La comprobación exploratoria externa posterior al congelamiento del modelo no se incluye como datos o figura en esta edición pública.

## Resultados agregados locales (F31)

La selección anidada LOSO comparó 325 configuraciones de 11 familias con 15 pliegues externos por participante y cuatro pliegues internos agrupados. Para los tres tercios temporales, el procedimiento anidado obtuvo macro-F1 medio por participante 0,740 (IC 95 % bootstrap 0,689–0,787) y ROC-AUC macro OvR 0,941 (0,921–0,959). La regresión logística tuvo el mayor punto de macro-F1 entre selectores de familia fija (0,751), pero no se estableció superioridad ni un único modelo universal.

La estratificación post hoc separó las piernas registradas. La izquierda incluyó 14 grupos, 70 grabaciones y 210 segmentos; la derecha, 15 grupos, 68 grabaciones y 204 segmentos. Catorce grupos fueron pareados. Macro-F1 fue 0,776 (IC 95 % 0,675–0,874) a izquierda y 0,686 (0,611–0,762) a derecha; la diferencia pareada izquierda menos derecha fue +0,092 (−0,052 a +0,238), que incluye cero. La cobertura por serie se resume en `reports/LATERALITY_F31_20261018.md` y el heatmap agregado. Estas métricas describen clasificación de posición temporal, no fatiga ni asimetría fisiológica.

En ventanas de 1 s con 50 % de solapamiento, la rama HP20 dio pendientes medianas por grupo de +19,30 cuentas ADC/min para RMS (IC95 % +12,06 a +41,41), −52,49 Hz/min para MDF (−68,38 a −8,54) y −44,03 Hz/min para MNF (−86,01 a −22,17). Las pendientes espectrales mantuvieron dirección descendente al repetir el cálculo sin solapamiento y en las otras ramas. La linealidad fue baja (R² mediano: RMS 0,031; MDF 0,118; MNF 0,083) y las trayectorias no fueron monotónicas. La amplitud se expresa en cuentas ADC, no en mV calibrados.

## Procesamiento comparado

Se compararon: señal sin filtrar; pasa-altas Butterworth IIR de cuarto orden a 20 y 40 Hz; pasa-banda IIR Butterworth de cuarto orden 20–400 Hz; y pasa-banda FIR de 101 coeficientes con ventana Hamming de 20–400 Hz. Los filtros se aplicaron en fase cero dentro de tramos válidos; gaps mayores de 1,5 ms se separaron sin interpolación. No se aplicó notch, porque el cribado espectral documentado no detectó una interferencia de red que superara el criterio preestablecido de 10 dB. Sin una referencia limpia no se puede declarar un filtro ganador ni medir SNR real.

Para la señal centrada de cada ventana, RMS = sqrt((1/N) sum x_c[n]^2). Se estimó la potencia espectral con periodograma unilateral Hann; MNF = sum(f_k P_k)/sum(P_k), y MDF es la frecuencia donde la potencia acumulada 20–400 Hz alcanza 50 %. La pendiente temporal se estimó por mínimos cuadrados y se resume por grupo con intervalos bootstrap a nivel de grupo. Una pendiente MDF/MNF negativa solo es compatible con el patrón clásico bajo contracciones sostenidas y fuerza aproximadamente controlada; aquí faltan fuerza/torque sincronizada, sitio anatómico confirmado, fase de repetición y referencia independiente. El primer tercio no se asume como basal, y no se fijó umbral ni inicio/pico de fatiga.

## Estructura

- `src/`: 16 módulos de QC, filtros, características, benchmark, análisis por pierna y resguardo de rutas locales.
- `bibliografia/`: las 44 referencias consolidadas en dos archivos BibTeX, sin cambios.
- `datasets/README.md`: organización local segura; no incluye datos.
- `reports/`: informe IEEE público F31, resumen y heatmap agregado por pierna/serie, nota metodológica y auditoría del paquete.
- `manifest/SHA256SUMS`: hashes del paquete.

## Reproducibilidad

Se verificaron Python 3.12.14, NumPy 2.3.5 y pandas 3.0.1. El entorno mínimo y comandos de ayuda están en `requirements.txt` y en cada módulo. La inspección de ayuda no lee datos ni lanza entrenamiento. Los argumentos de datos y salida deben apuntar a almacenamiento autorizado fuera de este repositorio.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m compileall -q src
python -m unittest discover -s tests -v
python src\entrenar_proxy_temporal.py --help
```

Antes de entrenar, inspecciona el esquema y las marcas temporales, define la etiqueta operacional, verifica el manifiesto de particiones por participante, mantén las salidas privadas fuera del repositorio y no trates el tercio temporal como fatiga o como fallo técnico.

## Límites y uso

No se verificaron modalidad RAW/RECT/ENV, frecuencia física del equipo, calibración, sitio muscular, carga, pausas, cadencia o repeticiones mecánicas. La columna local `Voltage_mV` no está validada como voltaje físico. Por tanto, esta investigación describe tendencias del canal guardado y desempeño de una tarea proxy, no impacto de un grupo muscular específico ni fatiga confirmada.

No se agregó una licencia: no se han verificado los derechos para otorgarla. El acceso a datos sigue sujeto a autorización, consentimiento y aprobación ética aplicable.
