# Auditoría del paquete público EMG

**Fecha:** 17 de octubre de 2026  
**Estado:** candidato seguro auditado localmente; la publicación remota se verifica por separado.

## Contenido y exclusiones

El paquete contiene 14 módulos Python portables, dos bibliografías consolidadas, instrucciones, una nota metodológica y la edición pública del informe IEEE en DOCX/PDF. No incluye archivos CSV/XLSX/NPY/NPZ, señales locales o externas, metadatos de participantes, características, etiquetas, particiones, predicciones, métricas por persona/pliegue/canal, recibos privados, modelos serializados ni figuras de casos. Se excluyó `participants.csv` y no se usó para el análisis.

La edición pública del informe conserva solo los resultados agregados locales y las figuras locales agregadas. Se quitaron las trazas, métricas por canal, probabilidades y figura correspondientes a la comprobación externa de una sola grabación. El informe de revisión local conserva esos elementos. Ninguna señal de los 15 participantes está en este paquete.

## Controles verificados

- 14/14 módulos se compilaron y su ayuda `--help` terminó correctamente sin leer señales ni entrenar modelos.
- DOCX con 44 referencias consecutivas, tres tablas (incluido el esquema editable) y cuatro figuras agregadas más el membrete institucional; sin comentarios activos o checklist de envío.
- PDF exportado desde Word, 12 páginas renderizadas e inspeccionadas visualmente; sin figuras externas, desbordes ni tablas rotas.
- Las dos bibliografías se compararon byte a byte con los archivos consolidados del proyecto y permanecen intactas.
- No se encontraron rutas locales, credenciales o datos individuales en los archivos de texto del paquete.
- La métrica del modelo corresponde al proxy de tercios temporales, no a fatiga; no hay ground truth independiente local y no se define umbral fisiológico.

## Límites científicos

La comprobación externa se hizo después de congelar el clasificador y solo examinó el proxy temporal; no se distribuyen sus resultados por ensayo ni se presenta como validación fisiológica. Las señales locales no tienen fuerza/torque sincronizada, sitio muscular local confirmado, fase de repetición validada ni calibración física de amplitud. MDF/MNF descendentes son un patrón compatible bajo condiciones controladas; aquí sus pendientes no monotónicas y R² bajos no confirman fatiga.

No se añadió una licencia porque los derechos para otorgarla no se verificaron. La reproducción completa requiere acceso autorizado a las grabaciones locales.
