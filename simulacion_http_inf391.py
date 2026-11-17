"""SIMULACION DE SOLICITUDES HTTP SOSPECHOSAS - INF-391.

Un solo script Python, ordenado en OCHO pasos, para explicar en clase.

X1: intervalo entre llegadas -> Exponencial con tasa variable por hora.
X2: categoria de solicitud -> Categorica.
X3: metodo HTTP -> Categorica, condicional a X2.
X4: respuesta HTTP 404 -> Bernoulli, condicional a X2 y X3.
Y : cantidad de respuestas 404 a solicitudes sospechosas en 72 horas.

La categoria "sospechosa" se obtuvo de indicadores heurísticos en el log;
no equivale a una intrusion confirmada. HTTP 404 tampoco equivale a bloqueo.

IMPORTANTE: la prueba 04-06/10/2025 es EXPLORATORIA/RETROSPECTIVA:
esas fechas ya se examinaron durante el desarrollo de este proyecto.
Cada hora se usa solamente el pasado para pronosticar la siguiente hora.
NO es un pronostico de las 72 horas realizado de una sola vez.
"""

# ===========================================================================
# PASO 1. IMPORTAR LIBRERIAS Y DEFINIR LAS RUTAS
# ===========================================================================
import csv
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Permite crear graficos sin abrir ventanas.
import matplotlib.pyplot as plt

CARPETA = Path(__file__).resolve().parent
ARCHIVO_DATOS = CARPETA / "datos" / "eventos_X2.csv"
RESULTADOS = CARPETA / "resultados"


# ===========================================================================
# PASO 2. FIJAR PARAMETROS Y LEER LAS OBSERVACIONES (NO EDITAR PARA 'ACERTAR')
# ===========================================================================
SEMILLA = 391
REPLICAS = 10_000                    # Monte Carlo principal; reducir a 1000 para clase.
REPLICAS_SENSIBILIDAD = 1_000
FECHA_INICIAL = datetime(2025, 9, 24)
HORAS_ENTRENAMIENTO = 10 * 24         # 24/09 al 03/10: 238 solicitudes.
HORAS_EVALUACION = 3 * 24            # 04/10 al 06/10: 172 solicitudes.
VENTANA_X1 = 2                       # Ultimas 2 horas observadas.
PESO_X1_RECIENTE = 0.95              # Tasa horaria = .95 reciente + .05 historica.
VENTANA_PROBABILIDADES = 5 * 24      # Ultimos 5 dias para tipos, metodos y 404.
FUERZA_CATEGORIAS = 4.0              # Suavizado ante categorias poco frecuentes.
FUERZA_METODOS = 4.0
FUERZA_RESPUESTA = 5.0

# Las variaciones de carga y de respuesta SON EXPERIMENTOS HIPOTETICOS.
# La unica prueba frente a observaciones es el escenario x1, respuesta base.
FACTORES_CARGA = (1, 5, 10)
FACTORES_404 = {"Bajo": 0.8, "Base": 1.0, "Alto": 1.2}
TIPOS = ("inyeccion_sql", "archivo_sensible", "ejecucion_comandos", "xss")
METODOS = ("GET", "POST", "DELETE")


def leer_datos():
    """Leer las 410 solicitudes previamente clasificadas desde logs Apache."""
    with ARCHIVO_DATOS.open(encoding="utf-8-sig", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    # La marca de tiempo contiene +07:00. Todas las filas usan esa zona.
    eventos = []
    for f in filas:
        instante = datetime.fromisoformat(f["timestamp"]).replace(tzinfo=None)
        hora = int((instante - FECHA_INICIAL).total_seconds() // 3600)
        if f["categoria_principal"] not in TIPOS or f["metodo"] not in METODOS:
            raise ValueError(f"Tipo o metodo no reconocido: {f}")
        eventos.append((hora, TIPOS.index(f["categoria_principal"]),
                        METODOS.index(f["metodo"]), int(f["estado_http"] == "404")))
    eventos.sort(key=lambda e: e[0])
    datos = np.array(eventos, dtype=np.int64)
    if datos.ndim != 2 or datos.shape != (410, 4):
        raise ValueError("Se esperan 410 eventos con cuatro campos: hora, tipo, metodo, 404.")
    if np.any(datos[:, 0] < 0) or np.any(datos[:, 0] >= 312):
        raise ValueError("Las fechas del CSV no corresponden al periodo documentado.")
    horas = np.bincount(datos[:, 0], minlength=312)
    respuestas = np.bincount(datos[:, 0], weights=datos[:, 3], minlength=312).astype(int)
    assert horas[:240].sum() == 238 and horas[240:].sum() == 172
    assert respuestas[:240].sum() == 205 and respuestas[240:].sum() == 129
    return datos, horas, respuestas


# ===========================================================================
# PASO 3. GENERAR X1, X2, X3 Y X4 POR TRANSFORMADA INVERSA / BERNOULLI
# ===========================================================================
def x1_exponencial(u, tasa_hora):
    """X1 = -ln(1-U)/lambda, en HORAS; tasa de solicitudes por hora."""
    if tasa_hora <= 0:
        return np.full_like(np.asarray(u, dtype=float), np.inf, dtype=float)
    return -np.log1p(-u) / tasa_hora


def x_categorica(u, probabilidades):
    """Seleccionar el indice del primer intervalo de la CDF que contiene U."""
    return np.minimum(np.searchsorted(np.cumsum(probabilidades), u, side="right"),
                      len(probabilidades)-1)


def x4_bernoulli(u, p404):
    """Vale 1 si el servidor responde 404; 0 para cualquier otra respuesta."""
    return np.asarray(u) < np.asarray(p404)


# ===========================================================================
# PASO 4. ESTIMAR PARAMETROS CON DATOS ANTERIORES A CADA HORA
# ===========================================================================
def preparar_horas(datos, conteos, inicio, fin, fin_entrenamiento):
    """Obtener los parametros para cada hora SIN mirar esa hora ni el futuro.

    X1 usa las 2 horas previas y una media historica ya conocida.
    X2, X3 y X4 usan las 120 horas previas (ventana movil de 5 dias).
    Suavizado: mezclar proporciones de la ventana con informacion anterior.
    El mismo estimador se usa en la simulacion y en la expectativa teorica.
    """
    if fin_entrenamiento > inicio:
        raise ValueError("La calibracion debe acabar antes de la evaluacion.")
    train = datos[datos[:, 0] < fin_entrenamiento]
    if len(train) == 0:
        raise ValueError("No hay datos de entrenamiento.")
    historica = float(conteos[:fin_entrenamiento].mean())
    prior_tipos = np.bincount(train[:, 1], minlength=4) / len(train)
    prior_metodos = np.bincount(train[:, 2], minlength=3) / len(train)
    parametros = []
    for h in range(inicio, fin):
        tasa = PESO_X1_RECIENTE * conteos[h-VENTANA_X1:h].mean() + (1-PESO_X1_RECIENTE)*historica
        ventana = datos[(datos[:, 0] >= h - VENTANA_PROBABILIDADES) & (datos[:, 0] < h)]
        n = len(ventana)
        tipos = np.bincount(ventana[:, 1], minlength=4)
        p_tipo = (tipos + FUERZA_CATEGORIAS*prior_tipos) / (n + FUERZA_CATEGORIAS)
        # Distribucion condicional del metodo para cada tipo: P(X3|X2).
        p_metodo = np.zeros((4, 3))
        for t in range(4):
            sub = ventana[ventana[:, 1] == t]
            num = np.bincount(sub[:, 2], minlength=3)
            p_metodo[t] = (num + FUERZA_METODOS*prior_metodos) / (len(sub) + FUERZA_METODOS)
        # P(404 | tipo, metodo). Suavizamos primero por tipo y despues por par.
        # Esto evita asignar arbitrariamente 0% o 100% si hay muy pocos datos.
        p_global = (float(ventana[:, 3].sum()) + 1) / (n + 2) if n else float(train[:, 3].mean())
        p_tipo_404 = np.zeros(4)
        p_par_404 = np.zeros((4, 3))
        for t in range(4):
            sub = ventana[ventana[:, 1] == t]
            p_tipo_404[t] = (sub[:, 3].sum() + FUERZA_RESPUESTA*p_global) / (len(sub) + FUERZA_RESPUESTA)
            for m in range(3):
                par = sub[sub[:, 2] == m]
                p_par_404[t, m] = (par[:, 3].sum() + FUERZA_RESPUESTA*p_tipo_404[t]) / (len(par) + FUERZA_RESPUESTA)
        p404_global_modelada = float(np.sum(p_tipo[:, None] * p_metodo * p_par_404))
        assert 0 <= p404_global_modelada <= 1
        parametros.append({"hora": h, "lambda_h": float(tasa), "p_tipo": p_tipo,
                           "p_metodo": p_metodo, "p404": p_par_404,
                           "p404_global": p404_global_modelada,
                           "esperadas_404": float(tasa * p404_global_modelada)})
    return parametros


# ===========================================================================
# PASO 5. SIMULAR LAS SOLICITUDES Y LAS RESPUESTAS DE UNA JORNADA
# ===========================================================================
def simular_72_horas(parametros, replicas, semilla, carga=1, factor404=1, detalle=False):
    """Simular 72 horas en replicas independientes, con X1 Exponencial.

    Se generan los momentos de llegada de solicitudes dentro de cada hora;
    luego se generan X2, X3 y X4. Vectorizamos las replicas para ahorrar tiempo,
    sin reemplazar X1 por conteos Poisson directamente.
    """
    rng = np.random.default_rng(semilla)
    totales = np.zeros(replicas, dtype=int)
    por_dia = np.zeros((replicas, 3), dtype=int)
    promedios_hora = np.zeros(len(parametros))
    detalles = []
    for i, par in enumerate(parametros):
        lam = par["lambda_h"] * carga
        prob_tipo = par["p_tipo"]
        cdf_metodo = np.cumsum(par["p_metodo"], axis=1)
        p404 = np.clip(par["p404"] * factor404, 0, 1)
        eventos_404_hora = np.zeros(replicas, dtype=int)
        # Cada replica inicia su primera espera dentro de esta hora.
        proxima = x1_exponencial(rng.random(replicas), lam)
        activas = np.flatnonzero(proxima < 1.0)
        while activas.size:
            t = x_categorica(rng.random(activas.size), prob_tipo)
            # X3 se genera en funcion del tipo obtenido en X2.
            u_metodo = rng.random(activas.size)
            m = np.minimum(np.sum(u_metodo[:, None] >= cdf_metodo[t], axis=1), 2)
            es_404 = x4_bernoulli(rng.random(activas.size), p404[t, m])
            eventos_404_hora[activas] += es_404
            if detalle and i < 8:  # Detalle de la primera replica en las primeras 8 h.
                pos = np.flatnonzero(activas == 0)
                if pos.size and len(detalles) < 100:
                    j = int(pos[0])
                    detalles.append({
                        "hora_indice": par["hora"],
                        "minuto_en_hora": round(float(proxima[0]*60), 4),
                        "X2_categoria": TIPOS[t[j]], "X3_metodo": METODOS[m[j]],
                        "X4_es_404": int(es_404[j]),
                    })
            # X1 siguiente: reiniciar el intervalo desde esta llegada.
            proxima[activas] += x1_exponencial(rng.random(activas.size), lam)
            activas = activas[proxima[activas] < 1.0]
        totales += eventos_404_hora
        por_dia[:, i//24] += eventos_404_hora
        promedios_hora[i] = eventos_404_hora.mean()
    return totales, por_dia, promedios_hora, detalles


# ===========================================================================
# PASO 6. VERIFICAR ECUACIONES Y EJECUTAR MONTE CARLO
# ===========================================================================
def comprobar_generadores():
    assert np.isclose(x1_exponencial(np.array([1-np.exp(-1)]), 2)[0], 0.5)
    assert [int(x_categorica(u, [0.2, 0.3, 0.5])) for u in (0.1, 0.25, 0.9)] == [0, 1, 2]
    assert [bool(x4_bernoulli(u, 0.7)) for u in (0.1, 0.8)] == [True, False]


def evaluar_modelo():
    datos, observados, reales_404 = leer_datos()
    comprobar_generadores()
    inicio = HORAS_ENTRENAMIENTO
    fin = inicio + HORAS_EVALUACION
    params = preparar_horas(datos, observados, inicio, fin, inicio)
    esperado = float(sum(p["esperadas_404"] for p in params))
    esperado_solicitudes = float(sum(p["lambda_h"] for p in params))
    valor_real = int(reales_404[inicio:fin].sum())
    # El detalle y las 10.000 replicas se ejecutan solo UNA vez para el escenario base.
    base = simular_72_horas(params, REPLICAS, np.random.SeedSequence([SEMILLA, 1, 1]), detalle=True)
    muestras, por_dia, medias_hora, detalle = base
    media_simulada = float(muestras.mean())
    error = 100 * abs(valor_real - media_simulada) / valor_real
    error_esperado = 100 * abs(valor_real - esperado) / valor_real
    # El estimador teorico debe aproximarse a la media Monte Carlo.
    stderr = float(muestras.std(ddof=1) / np.sqrt(REPLICAS))
    if not abs(esperado - media_simulada) < 5*stderr + 0.1:
        raise AssertionError("La simulacion no concuerda con la esperanza matematica.")
    observados_dia = reales_404[inicio:fin].reshape(3, 24).sum(axis=1)
    filas_diarias = []
    for k in range(3):
        estimado_dia = float(por_dia[:, k].mean())
        real_dia = int(observados_dia[k])
        filas_diarias.append({
            "fecha": (FECHA_INICIAL + timedelta(hours=inicio + k*24)).strftime("%Y-%m-%d"),
            "404_observadas": real_dia, "404_simuladas_media": round(estimado_dia, 4),
            "error_relativo_pct": round(100*abs(real_dia-estimado_dia)/real_dia, 4) if real_dia else "no definido",
        })
    filas_horarias = []
    for i, p in enumerate(params):
        hora = FECHA_INICIAL + timedelta(hours=p["hora"])
        filas_horarias.append({"hora_local": hora.strftime("%Y-%m-%d %H:00"),
            "solicitudes_observadas": int(observados[p["hora"]]),
            "404_observadas": int(reales_404[p["hora"]]),
            "lambda_solicitudes_h": round(p["lambda_h"], 6),
            "prob_404_condicional_promedio": round(p["p404_global"], 6),
            "404_esperadas_modelo": round(p["esperadas_404"], 6),
            "404_simuladas_media": round(float(medias_hora[i]), 6)})
    # Escenarios hipoteticos: 3 intensidades * 3 factores de probabilidad.
    # Base reusa sus replicas; los otros escenarios son independientes.
    sensibilidad = []
    for a, carga in enumerate(FACTORES_CARGA):
        for b, (nombre, factor404) in enumerate(FACTORES_404.items()):
            if carga == 1 and nombre == "Base":
                replicas_resultado = muestras
            else:
                replicas_resultado = simular_72_horas(
                    params, REPLICAS_SENSIBILIDAD,
                    np.random.SeedSequence([SEMILLA, 100+a, 200+b]),
                    carga, factor404)[0]
            sensibilidad.append({"carga": f"x{carga}", "respuesta_404": nombre,
                                 "404_simuladas_media_72h": round(float(replicas_resultado.mean()), 4),
                                 "replicas": len(replicas_resultado), "datos_reales": "NO; escenario hipotetico"})
    return {"datos": datos, "observados": observados, "reales_404": reales_404,
            "params": params, "muestras": muestras, "medias_hora": medias_hora,
            "filas_diarias": filas_diarias, "filas_horarias": filas_horarias,
            "sensibilidad": sensibilidad, "detalle": detalle,
            "valor_real": valor_real, "media_simulada": media_simulada,
            "esperado": esperado, "esperado_solicitudes": esperado_solicitudes,
            "error_pct": error, "error_esperado_pct": error_esperado,
            "stderr": stderr}


# ===========================================================================
# PASO 7. EXPORTAR TABLAS CSV Y GRAFICOS
# ===========================================================================
def guardar_csv(nombre, filas):
    if not filas:
        return
    with (RESULTADOS / nombre).open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def graficar(r):
    fechas = [f["fecha"] for f in r["filas_diarias"]]
    reales = [f["404_observadas"] for f in r["filas_diarias"]]
    simulados = [f["404_simuladas_media"] for f in r["filas_diarias"]]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x-0.2, reales, 0.4, label="404 observadas")
    ax.bar(x+0.2, simulados, 0.4, label="404 simuladas (media Monte Carlo)")
    ax.set_xticks(x, fechas); ax.set_ylabel("Respuestas HTTP 404")
    ax.set_title("Contraste exploratorio: resultados diarios")
    ax.legend(); fig.tight_layout()
    fig.savefig(RESULTADOS/"01_respuestas_404_diarias.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot([f["404_observadas"] for f in r["filas_horarias"]], label="Observado", marker=".")
    ax.plot(r["medias_hora"], label="Simulado: promedio por hora")
    ax.set_xlabel("Hora durante los tres dias de evaluacion")
    ax.set_ylabel("Cantidad de respuestas 404")
    ax.set_title("Contraste horario: variabilidad y rafagas")
    ax.legend(); fig.tight_layout()
    fig.savefig(RESULTADOS/"02_respuestas_404_horarias.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(r["muestras"], bins=45)
    ax.axvline(r["valor_real"], linestyle="--", label="Observado: 129")
    ax.axvline(r["media_simulada"], linestyle=":", label="Media simulada")
    ax.set_xlabel("404 simuladas por replica (72 h)")
    ax.set_ylabel("Frecuencia de replicas")
    ax.set_title("Distribucion Monte Carlo del resultado Y")
    ax.legend(); fig.tight_layout()
    fig.savefig(RESULTADOS/"03_histograma_montecarlo.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    for nombre in FACTORES_404:
        puntos = [f for f in r["sensibilidad"] if f["respuesta_404"] == nombre]
        ax.plot([1, 5, 10], [f["404_simuladas_media_72h"] for f in puntos],
                marker="o", label=f"Prob. 404 {nombre}")
    ax.set_xlabel("Factor hipotetico de llegadas X1")
    ax.set_ylabel("Respuestas 404 simuladas en 72 h")
    ax.set_title("Analisis de sensibilidad (NO es validacion)")
    ax.legend(); fig.tight_layout()
    fig.savefig(RESULTADOS/"04_sensibilidad.png", dpi=150); plt.close(fig)


# ===========================================================================
# PASO 8. APLICAR FORMULA DEL DOCENTE E INTERPRETAR LA VALIDACION
# ===========================================================================
def main():
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    r = evaluar_modelo()
    guardar_csv("comparacion_diaria.csv", r["filas_diarias"])
    guardar_csv("comparacion_horaria.csv", r["filas_horarias"])
    guardar_csv("escenarios_sensibilidad.csv", r["sensibilidad"])
    guardar_csv("ejemplo_eventos.csv", r["detalle"])
    guardar_csv("replicas_montecarlo.csv", [
        {"replica": i+1, "respuestas_404": int(n)}
        for i, n in enumerate(r["muestras"])])
    graficar(r)

    print("\nMODELADO Y SIMULACION HTTP - INF-391 | UN SOLO SCRIPT")
    print("="*67)
    print("PASOS 1-5. 4 variables: X1 Exponencial, X2/X3 Categorica, X4 Bernoulli")
    print(f"PASO 6. Monte Carlo: {REPLICAS:,} replicas base y {REPLICAS_SENSIBILIDAD:,} por escenario alternativo (9 en total)")
    print("PASO 7. CSV y graficos guardados en:", RESULTADOS)
    print("\nPASO 8. COMPARACION CON DATOS OBSERVADOS (04-06 octubre 2025)")
    print("Entrenamiento: 24/09 al 03/10 (238 solicitudes, 205 respuestas 404)")
    print("Evaluacion exploratoria: 04/10 al 06/10 (172 solicitudes, 129 respuestas 404)")
    print(f"Solicitudes esperadas en linea: {r['esperado_solicitudes']:.2f}")
    print(f"Respuestas 404 reales = {r['valor_real']}")
    print(f"Respuestas 404 simuladas (media) = {r['media_simulada']:.4f}")
    print(f"Respuestas 404 esperadas por ecuaciones = {r['esperado']:.4f}")
    print(f"Error relativo = |{r['valor_real']} - {r['media_simulada']:.4f}| / {r['valor_real']} * 100 = {r['error_pct']:.2f} %")
    print(f"IC95 de la media de replicas (aprox.): [{r['media_simulada']-1.96*r['stderr']:.3f}, "
          f"{r['media_simulada']+1.96*r['stderr']:.3f}]")
    for fila in r["filas_diarias"]:
        print("  ", fila)
    print("\nADVERTENCIA: NO es validacion independiente. El periodo ya fue explorado")
    print("al desarrollar el proyecto. Las estimaciones se actualizan cada hora")
    print("con datos reales ANTERIORES; no se predicen 72 horas desde el inicio.")
    print("Una fuente nueva es necesaria para validacion externa definitiva.")
    if r["error_pct"] <= 10:
        print("CUMPLE numericamante el umbral del 10 % EN ESTE CONTRASTE EXPLORATORIO.")
    else:
        print("NO CUMPLE el umbral del 10 % EN ESTE CONTRASTE EXPLORATORIO.")


if __name__ == "__main__":
    main()
