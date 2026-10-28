# F60 — demo local para RAR con series `set01`–`set05`

**Fecha:** 2026-10-28
**Estado:** lanzador implementado; prueba funcional local terminada; salidas sujetas a privacidad.

## Qué añade

- `INICIAR_DEMO_RAW_RAR.bat` inicia el demo con `Downloads/raw.rar` por defecto o permite pasar otra ruta.
- `src/ejecutar_demo_raw_rar.py` lee un RAR compatible usando `tar.exe`, valida nombres y cabeceras y genera un ZIP temporal con nombres pseudonimizados. No extrae miembros usando rutas del archivo y no modifica el RAR original.
- Nombres `Pxx_set01-05_dr/iz.csv` se traducen a `P01/P02/.../dr_R1-5.csv` o `iz_R1-5.csv`. Se exige una matriz completa de cinco series y ambos lados por cada código del RAR.
- La ejecución pasa ese ZIP temporal al procesador descriptivo CSV/ZIP y genera `index.html`, tableros individuales y tablas CSV bajo la carpeta local de resultados. El ZIP temporal se elimina al terminar.

## Verificación

- 4 pruebas específicas verifican la conversión de `set01`–`set05`, el pseudónimo, rechazo de matrices incompletas, nombres inseguros y duplicados; las 4 aprobaron.
- `python -m unittest discover -s tests -v`: **117 pruebas aprobadas** en la verificación completa actual.
- El flujo fue ejecutado localmente con el RAR proporcionado; el archivo y los productos derivados permanecieron fuera del repositorio. El inventario, señales, tableros y métricas individuales no se incluyen en esta nota pública.
- La página local confirma en su resumen que no se cargó un modelo y que no se emitieron predicciones de repeticiones o fatiga. Las pruebas de software y la ejecución descriptiva no constituyen validación fisiológica.
- El índice HTML actualizado ofrece una sola vista con la línea temporal por grabación, un resumen agregado por serie/lateralidad y una tabla filtrable de tiempos de inicio y fin. Usa el reloj local de cada archivo, no crea una cronología sincronizada entre grabaciones y no enlaza a tableros individuales.

## Límites

El mapeo de set a R se basa en la numeración de nombres. No demuestra que cada evento de señal sea una repetición mecánica. Los eventos detectados son candidatos de actividad y el procesador no predice fatiga, carga, fuerza ni nivel fisiológico. Para RAR distintos se requiere el mismo patrón exacto y un `tar.exe` de Windows que pueda leer el formato RAR concreto.

El repositorio no incluye el RAR original, CSV de participantes, páginas HTML individuales, resultados por participante ni checkpoint privado.
