# Pipeline reproducible Cerqueira

Este código reproduce el análisis del dataset público Cerqueira para la tarea de **fatiga autopercibida ordinal del miembro superior (clases 0/1/2)**. No mide pérdida de fuerza, no etiqueta fatiga fisiológica del cuádriceps y no es una recomendación de despliegue: en la validación LOSO disponible, el baseline de tiempo superó al clasificador EMG.

## Privacidad y archivos de salida

El repositorio contiene código y pruebas sintéticas; no contiene los ZIP de señal o etiquetas, tablas de características, particiones, predicciones, recibos por ensayo ni modelos. Las ejecuciones generan información por participante, ensayo o ventana y artefactos de modelo. Guárdalos en un directorio restringido **fuera del checkout, de carpetas sincronizadas y de cualquier repositorio público**. No añadas esos productos a Git.

Los comandos rechazan rutas de salida coincidentes, existentes, dirigidas a sus entradas/modelos o ubicadas dentro del checkout público. Usa una carpeta privada nueva para cada ejecución y comprueba que no esté sincronizada con un servicio compartido. Las pruebas de esta guía utilizan datos sintéticos; no reproducen las cifras del estudio.

## Entorno

Probado con Python 3.12.14, NumPy 2.3.5 y pandas 3.0.1, fijados en `requirements.txt` del repositorio. Desde la raíz del checkout:

```powershell
python -m pip install -r requirements.txt
python src\cerqueira\auditar_cerqueira_emg_label_alignment.py --help
python src\cerqueira\extraer_caracteristicas_cerqueira.py --help
python src\cerqueira\entrenar_cerqueira_loso.py --help
python src\cerqueira\inferir_cerqueira_trial.py --help
```

El proyecto no descarga datos. Obtén los archivos únicamente desde la fuente oficial y conforme a sus condiciones de acceso. El auditor verifica por defecto los MD5 publicados para los dos ZIP oficiales; los valores pueden cambiarse explícitamente solo para una auditoría distinta y documentada.

## Flujo completo

Elige de antemano una carpeta local restringida; el siguiente ejemplo es una ruta ilustrativa:

```powershell
$privateOutput = 'D:\restricted\cerqueira-run-01'
New-Item -ItemType Directory -Force $privateOutput | Out-Null
```

1. Audita integridad, correspondencia de ensayos y soporte temporal. Conserva JSON y CSV en la carpeta restringida:

```powershell
python src\cerqueira\auditar_cerqueira_emg_label_alignment.py `
  'D:\restricted\source\sEMG_data_full_official.zip' `
  'D:\restricted\source\self_perceived_fatigue_index.zip' `
  --output-json "$privateOutput\alignment.json" `
  --output-csv "$privateOutput\alignment_trials.csv"
```

2. Extrae características por ventanas usando solo los archivos cuya correspondencia quedó auditada:

```powershell
python src\cerqueira\extraer_caracteristicas_cerqueira.py `
  'D:\restricted\source\sEMG_data_full_official.zip' `
  'D:\restricted\source\self_perceived_fatigue_index.zip' `
  "$privateOutput\alignment.json" `
  --output-csv "$privateOutput\features.csv" `
  --summary-json "$privateOutput\features_summary.json"
```

3. Ejecuta la evaluación anidada LOSO agrupada por participante. Las predicciones OOF, métricas detalladas, recibos y modelos por pliegue son privados:

```powershell
python src\cerqueira\entrenar_cerqueira_loso.py `
  "$privateOutput\features.csv" `
  "$privateOutput\features_summary.json" `
  "$privateOutput\loso" `
  --alignment-receipt "$privateOutput\alignment.json"
```

4. Para la reproducción de un pliegue retenido o un análisis autorizado, proporciona un CSV compatible y el **modelo de pliegue** que lo excluyó, junto con el manifiesto de hashes correspondiente. La inferencia no lee etiquetas:

```powershell
python src\cerqueira\inferir_cerqueira_trial.py `
  'D:\restricted\source\trial.csv' `
  --model "$privateOutput\loso\fold_models\heldout_subject_13.npz" `
  --fold-manifest "$privateOutput\loso\fold_models_manifest.json" `
  --output-csv "$privateOutput\heldout_scores.csv" `
  --summary-json "$privateOutput\heldout_inference.json"
```

Los CSV deben contener cuatro canales EMG y cuatro columnas de tiempo sincronizadas con encabezados compatibles con `x [s]`. La entrada debe ser finita, monótona y continua; los fragmentos, si se usan, deben entregarse en orden y sin huecos ni solapamientos. El cargador valida el hash del modelo contra el manifiesto, el orden de características, las dimensiones y la finitud de las matrices. Ese hash comprueba que los dos archivos coinciden; **no autentica quién los creó**. Obtén el modelo y su manifiesto por una vía de confianza. Los modelos de pliegue son para evaluación fuera de participante y no equivalen a un artefacto universal de despliegue.

La CLI también acepta el artefacto final full-fit de `entrenar_cerqueira_loso.py` con `--final-manifest <final_model_manifest.json>`, en lugar de `--fold-manifest`. El cargador exige el SHA-256 y el contrato de preprocesamiento, características, filtros, muestreo, clase y dimensiones declarados en el manifiesto. El extractor admite ventanas completas de 1 s sin solapamiento; un manifiesto que declare `window_overlap_fraction` distinto de cero se rechaza porque no coincide con la extracción implementada. Los manifiestos son mutuamente excluyentes. El paquete público no contiene un artefacto final; proporciona el modelo y el manifiesto localmente y mantenlos fuera del checkout. La inferencia resultante sigue estimando únicamente las etiquetas de auto-reporte ordinal de miembro superior.

Los lectores fallan de forma cerrada al exceder límites de recursos: 8 GiB por archivo ZIP externo, 10.000 miembros, 512 MiB por miembro, 8 GiB expandidos por archivo y razón de compresión máxima de 10.000; los CSV de señal y etiquetas tienen un límite de 1.000.000 de filas. El entrenador limita la tabla de características a 512 MiB y 250.000 filas, y su resumen JSON a 4 MiB; calcula el SHA-256 de la tabla por bloques, sin copiar el archivo completo a memoria. Los modelos NPZ admiten hasta 64 MiB comprimidos o expandidos, 32 miembros y 512 columnas de características; los manifiestos de pliegue se limitan a 4 MiB. Estos límites cubren el corpus oficial verificado y deben revisarse si se usa otra fuente.

La consola presenta un resumen sin nombres de archivos, hashes ni IDs de sujeto. Los recibos JSON y CSV detallados conservan información de procedencia y deben permanecer restringidos fuera del checkout.

## Verificación del código

Desde la raíz del repositorio:

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

`tests/test_cerqueira_public_pipeline.py` genera ZIP anidados, señales, etiquetas y modelos sintéticos en directorios temporales. Comprueba alineación, extracción, entrenamiento agrupado, inferencia CSV por ventana, montaje de fragmentos, hashes y rechazo de sobrescritura, colisiones y rutas dentro del checkout. No lee los ZIP oficiales.
