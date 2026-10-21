# Actualización pública F54 — 21 de octubre de 2026

Base: F53, commit 0fc2ce1522d15ac15ad9560760794a69d498d26e.

Se añadieron el guion de presentación y dos CSV agregados de la sección 9: 36 filas de métricas por tarea/modelo/configuración y 36 matrices de confusión agregadas. El README incorpora los resultados resumidos y conserva intacto su bloque “Límites y uso”.

La verificación local de esta rama pasó compileall y 91/91 pruebas. El manifiesto enumera los archivos según los blobs Git y se valida antes del commit.

F55, posición temporal local: Macro-F1 0,7163869 ± 0,1464587; exactitud 0,7362963 ± 0,1283779; ROC-AUC macro OvR 0,9170451 ± 0,0670841; PR-AUC macro 0,8704126 ± 0,0975770; 15 pliegues agrupados. La etiqueta son tercios de grabación, no fatiga. F31 conserva Macro-F1 histórico 0,740 para la misma tarea. La diferencia entre los cortes no se reconcilió en F54, por lo que ambos permanecen separados.

Borg (RPE ≥17, hombro) y Cerqueira (auto-reporte ordinal de miembro superior) continúan separados de la evaluación local de cuádriceps. Ninguna cifra de estas ramas valida fatiga del cuádriceps.

El guion tiene 10 secciones cronometradas, 1.238 palabras habladas y tiempos que suman 10:00; el ritmo indicado se corrigió a aproximadamente 124 palabras/minuto. El diseño editable Canva no se exporta aquí.

No se publicaron el archivo de resultados por pliegue (625 filas), manifiesto privado, señales, predicciones, características individuales, checkpoints ni el informe completo de curso, que contiene identidades de autor y valores por grupo de participante. El DOCX/PDF público F31 sigue siendo histórico.

El resumen contiene 26 configuraciones locales (15 pliegues cada una), 5 modelos Borg (34 pliegues cada uno) y 5 modelos Cerqueira (13 pliegues cada uno). Se comprobó que los CSV agregados no tienen campos de identidad, archivo, ensayo, pliegue, nombre o ruta local. El código F53 ya rechaza contratos de inferencia Cerqueira con solapamiento distinto de cero; esta actualización no cambia el código de procesamiento.

El paquete permanece intermedio. No se declara validación fisiológica de fatiga de cuádriceps ni inferencia Borg independiente reproducida.
