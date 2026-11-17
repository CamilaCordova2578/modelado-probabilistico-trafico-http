# Simulación INF-391 — un único script Python

**Tema:** Modelado y simulación del flujo de posibles intentos de intrusión y las respuestas HTTP en servidores web institucionales mediante métodos probabilísticos.

El proyecto utiliza **un único archivo Python**, `simulacion_http_inf391.py`, organizado en ocho pasos claramente comentados. El archivo `datos/eventos_X2.csv` es únicamente la **entrada con 410 solicitudes clasificadas** provenientes del análisis anterior de registros Apache de Organization X (las etiquetas son heurísticas). La carpeta `resultados/` contiene las salidas generadas automáticamente. **No hay otros módulos Python.**

## Ejecutar en Windows

Abre PowerShell en esta carpeta y ejecuta:

```powershell
py -m pip install numpy matplotlib
py simulacion_http_inf391.py
```

También sirve `python` si ese es el comando configurado en tu equipo.

## Los ocho pasos del código

1. Importar bibliotecas y localizar archivos.
2. Configurar el experimento y leer los 410 eventos observados.
3. Generar X1 Exponencial, X2 categórica, X3 categórica y X4 Bernoulli.
4. Estimar parámetros de cada hora usando solo observaciones anteriores.
5. Simular llegadas HTTP y sus respuestas.
6. Verificar las ecuaciones y repetir Monte Carlo.
7. Exportar CSV y cuatro gráficos.
8. Comparar el resultado final **Y** con las observaciones mediante error relativo porcentual.

## Definición de variables

- **X1:** tiempo entre solicitudes HTTP sospechosas. Exponencial por tramos horarios, con `lambda_h = 0.95 * promedio(conteo anterior 2 h) + 0.05 * media histórica entrenada`.
- **X2:** categoría: inyección SQL, archivo sensible, ejecución de comandos o XSS. Categórica.
- **X3:** método HTTP: GET, POST o DELETE. Categórica condicional a X2.
- **X4:** indicador de respuesta **HTTP 404**, 1 sí y 0 no. Bernoulli condicional a X2 y X3.
- **Y:** número total de respuestas 404 en 72 horas dentro de las solicitudes clasificadas como sospechosas. Se estima promediando las réplicas Monte Carlo.

La distribución de X2, la distribución condicional de X3 y la probabilidad condicional de X4 se actualizan con las observaciones de los **cinco días anteriores**. Se emplean pseudoconteos para evitar probabilidades 0/1 infundadas en grupos raros.

## Datos de evaluación y límites

- Entrenamiento: 24/09–03/10 de 2025: 238 solicitudes, 205 respuestas 404.
- Evaluación exploratoria: 04/10–06/10 de 2025: 172 solicitudes, 129 respuestas 404.
- Modelo original exploratorio: 8,03 % de error del total en otra configuración.
- **Esta nueva implementación:** 10.000 réplicas base, 137,4239 respuestas 404 simuladas en promedio frente a 129 observadas, **6,53 %** de error relativo sobre el total. El resultado puede cambiar ligeramente si modificas la semilla, parámetros o número de réplicas.
- Importante: **NO es validación independiente final**. El período 04–06/10 ya se examinó en la construcción del proyecto; se muestra como contraste retrospectivo. No se debe afirmar que el modelo alcanzó validez empírica final sin evaluarlo con registros nuevos.
- Son **72 pronósticos de una hora**, actualizados tras conocer las horas anteriores; no una predicción de 72 horas cerrada de una sola vez.
- El error acumulado puede ser bajo aunque un día concreto supere el 10 %. Presentar también `comparacion_diaria.csv` y `comparacion_horaria.csv`.
- Los escenarios de carga ×1, ×5, ×10 y variaciones de probabilidad 404 son **experimentos hipotéticos de sensibilidad**; no se comparan con observaciones.
- Una solicitud con indicadores de intrusión no es un ataque confirmado; una respuesta 404 significa recurso no encontrado y no es prueba de bloqueo de seguridad.

## Fórmula de validación de la salida (etapa de Pruebas)

`Error (%) = abs(Y_real - media_Y_simulada) / Y_real * 100`.

Es la fórmula de la clase. La salida Y **no es** el intervalo X1 ni la espera de un analista; ahora representa las respuestas 404 en un período de 72 horas.

## Archivos de resultados

- `comparacion_diaria.csv` y `01_respuestas_404_diarias.png`: comparación por día.
- `comparacion_horaria.csv` y `02_respuestas_404_horarias.png`: comparación por hora.
- `replicas_montecarlo.csv` y `03_histograma_montecarlo.png`: distribución del resultado final Y entre las 10.000 réplicas.
- `escenarios_sensibilidad.csv` y `04_sensibilidad.png`: experimentos hipotéticos.
- `ejemplo_eventos.csv`: primeros eventos generados con X2, X3 y X4; útil para explicarlo al grupo.
