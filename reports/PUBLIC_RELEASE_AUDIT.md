# Auditoría del paquete público EMG — F31

**Fecha:** 18 de octubre de 2026
**Estado:** candidato público agregado actualizado; la publicación remota se verificará contra el commit de cierre.

## Contenido y exclusiones

El paquete contiene 16 módulos Python, dos bibliografías consolidadas, instrucciones, el informe IEEE público F31 en DOCX/PDF, el heatmap y el resumen agregado por pierna/serie. No contiene archivos de participantes, señales locales o externas, métricas por persona/pliegue/canal, etiquetas, características, particiones, predicciones, recibos privados, modelos serializados ni figuras de caso. `participants.csv` se excluyó y no se usó para esta evaluación.

El informe público conserva resultados agregados locales. Se eliminaron las figuras de R0/R0_1, los puntos de rendimiento por participante y la señal de una grabación individual; también se borraron del DOCX los medios incrustados sin uso. Las 44 referencias consolidadas permanecen sin cambios.

## Controles verificados

- Los 16 módulos compilan; los módulos de lateralidad y filtros rechazan entradas o salidas ubicadas dentro del checkout público.
- Las pruebas verifican agregación por persona, heatmap izquierda/derecha y controles de ruta; `--help` no lee grabaciones ni entrena.
- DOCX con 44 referencias, 15 tablas nativas y ocho figuras agregadas o esquemas editables; sin checklist de envío, placeholders ni contenido individual de las tres figuras retiradas.
- PDF exportado desde Word en A4, con 22 páginas renderizadas e inspeccionadas; no se detectaron desbordes ni tablas rotas.
- Las dos bibliografías se cotejaron con las versiones consolidadas y no se modificaron.
- La matriz por lado reporta 14 grupos izquierdos y 15 derechos, con 14 pareados; la diferencia Macro-F1 incluye cero.

## Alcance científico

La variable objetivo es el tercio temporal temprano, medio o tardío de cada grabación. Las métricas por pierna describen esa clasificación temporal y no demuestran fatiga fisiológica ni asimetría neuromuscular. No hay etiquetas independientes de fatiga, fuerza/torque sincronizada, sitio anatómico local confirmado ni calibración física de amplitud; no se define umbral, inicio o pico de fatiga. Esta entrega F31 se limita al análisis local de los 15 grupos y no añade validación externa.

No se añadió una licencia porque los derechos para otorgarla no se verificaron. La reproducción completa requiere acceso autorizado a las grabaciones locales.
