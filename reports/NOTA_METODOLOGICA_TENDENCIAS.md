# Nota metodológica pública de tendencias RMS, MDF y MNF

## Alcance

Se analizaron las grabaciones locales de los 15 grupos derivados de códigos de archivo. `participants.csv` se usó para reconciliación administrativa del proyecto, pero sus campos no fueron entradas del análisis de señales/características ni covariables. R1–R5 (138 archivos) sustentan tendencias; R0/R0_1 (2) se conservaron solo para control de calidad. Se mantuvieron juntos los archivos de cada persona. La evaluación F31 por pierna reutilizó las predicciones anidadas congeladas y está resumida en `LATERALITY_F31_20261018.md`; no hubo ajuste de modelo por lado.

## Filtros y ventanas

Comparadores: sin filtrar; Butterworth IIR pasa-altas 20 Hz y 40 Hz; Butterworth IIR pasa-banda 20–400 Hz; FIR pasa-banda 20–400 Hz, 101 coeficientes, Hamming. IIR de fase cero por tramo. Gaps mayores de 1,5 ms separan tramos; no se interpola ni atraviesan discontinuidades. Notch 50/60 Hz no aplicado porque no se detectó pico que superara el cribado previo de 10 dB. Ventana primaria: 1 s, paso 0,5 s; sensibilidad: 1 s sin solapamiento.

## Definiciones

Para x_c[n] = x[n] − media(x), RMS = sqrt((1/N) Σ x_c[n]²). El periodograma unilateral Hann estima P_k. En banda fija 20–400 Hz, MNF = Σ(f_k P_k)/ΣP_k, y MDF satisface Σ_(f_k≤MDF) P_k = 0,5ΣP_k. La pendiente se calcula por mínimos cuadrados y tiempo en minutos: β₁ = Σ[(t−t̄)(y−ȳ)] / Σ(t−t̄)². El cambio porcentual entre dos fases comparables sería ΔMDF% = 100(MDF_final − MDF_inicial)/MDF_inicial y, de forma análoga, ΔMNF%. No se calcula aquí porque el primer 10 % del archivo no es un basal confirmado. Las incertidumbres agregadas se obtienen con bootstrap por grupo/participante. R² indica ajuste lineal, no confirma fatiga.

## Resultados de tendencia

En HP20 y 50 % de solapamiento, mediana de pendientes por grupo: RMS +19,30 cuentas ADC/min (IC95 % +12,06 a +41,41; R² 0,031); MDF −52,49 Hz/min (−68,38 a −8,54; R² 0,118); MNF −44,03 Hz/min (−86,01 a −22,17; R² 0,083). La dirección MDF/MNF persistió sin solapamiento y en otras ramas, pero las trayectorias tuvieron retrocesos y finales no monotónicos.

## Interpretación y umbral

MDF/MNF descendentes son compatibles con compresión espectral durante contracciones isométricas sostenidas a fuerza aproximadamente constante; no basta una pendiente negativa para identificar fatiga. RMS puede subir, estabilizarse o bajar y no es criterio único. Los registros carecen de fuerza/torque sincronizada, fase de repetición y sitio anatómico confirmado. La modalidad del canal y el estado basal son desconocidos; el primer tercio no es una referencia sin fatiga. No existe un porcentaje o umbral universal. Un umbral o cambio de punto calculado sobre estos mismos indicadores sería exploratorio y circular, por lo que no se reportó inicio/pico ni nivel de fatiga.

La comparación de filtros solo mide sensibilidad digital: sin una señal limpia de referencia, no se estima SNR ni se selecciona un filtro óptimo. Las señales se guardaron en cuentas ADC; `Voltage_mV` no posee calibración física documentada.
