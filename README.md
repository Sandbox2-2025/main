# Demostrador ferroviario IQM–DB: CPU e IQM en Q-Centroid

Este repositorio ejecuta dos algoritmos sobre **el mismo dataset simulado** de
10 servicios y 20 ciclos candidatos. El objetivo es comparar la calidad de la
solución clásica exacta con la obtenida por QAOA en un dispositivo IQM. Las
ciudades son reales; horarios y distancias del dataset son ficticios. El caso
sirve como demostrador, no como planificación operativa de Deutsche Bahn.

## Archivos en la raíz

| Archivo | Función |
| --- | --- |
| `qcentroid.py` | Punto de entrada obligatorio; dirige la ejecución según `solver_params.algorithm`. |
| `solver_cpu.py` | Valida el dataset y resuelve exactamente mediante programación dinámica sobre conjuntos de servicios. |
| `solver_iqm.py` | Valida el dataset, prepara el circuito QAOA, lo ejecuta en IQM y evalúa las muestras. |
| `requirements.txt` | Instala el adaptador Qiskit de IQM y Aer para simulación opcional. |

Q-Centroid invoca `run(input_data, solver_params, extra_arguments)` desde
`qcentroid.py`. Los otros dos módulos se importan desde ese archivo. Se requiere
Python 3.11 o posterior. El JSON se proporciona como `input_data` del job; no
es necesario guardarlo en el repositorio.

## Configuración de los jobs

El parámetro `algorithm` es **obligatorio**; no se elige el algoritmo en función
de la presencia de un token. Ejecute CPU e IQM como dos jobs sobre el mismo
dataset y conserve la revisión del código utilizada en cada ejecución.

### Referencia exacta en CPU

`solver_params`:

```json
{"algorithm": "cpu"}
```

No requiere credenciales ni paquetes externos. Calcula el conjunto compatible
con mayor suma de pesos. Si hay empate, prefiere menos ciclos (trenes). También
informa por separado el mejor plan que cubre los diez servicios.

### QAOA en IQM

`solver_params` de ejemplo:

```json
{
  "algorithm": "iqm",
  "qaoa_depth": 1,
  "shots": 4096,
  "gamma": 0.8,
  "beta": 0.35
}
```

El código actual espera `iqm_token` dentro de `extra_arguments`. Configure el
valor mediante el mecanismo seguro del entorno de ejecución; **no incorpore el
token al repositorio ni a este README**. `quantum_computer` es opcional y tiene
por defecto `emerald`. La URL se toma de `IQM_SERVER_URL` o, si no se define,
de `https://resonance.iqm.tech/`.

Para una prueba explícita en Aer, use `algorithm: "iqm"` y añada
`{"use_iqm": false}` en `extra_arguments`. En ese caso `backend_used` será
`AerSimulator`: **no es una ejecución en hardware IQM**. No existe un cambio
automático a Aer si falla IQM.

Los ángulos `gamma` y `beta` son valores fijos para cada job. Esta versión
prepara y mide un circuito QAOA, pero no incorpora un bucle clásico que
optimice los ángulos. Por ello, su calidad puede depender notablemente de los
parámetros elegidos y del número de muestras.

## Problema matemático y resultado esperado

Cada ciclo candidato tiene peso
`weight = 2 * passenger_km - empty_km`. En este JSON, `passenger_km` significa
kilómetros recorridos por el tren prestando servicio, **no** pasajeros-km
ponderados por ocupación. Dos ciclos que comparten un servicio están unidos por
una arista de conflicto y no pueden seleccionarse juntos.

CPU maximiza la suma de pesos de ciclos sin conflictos. IQM codifica el mismo
problema como
`H(x) = -sum(weight_i * x_i) + lambda * sum(x_i * x_j)` para las aristas, con
`lambda = 4 * max(weight_i)`. Después de medir, elimina conflictos de cada
muestra y prioriza una muestra de cobertura completa si la encuentra. Esa
reparación **no garantiza** que todos los servicios queden cubiertos.

La referencia exacta del dataset proporcionado es **C01 + C03**: 10 de 10
servicios cubiertos una sola vez, 2 trenes, 4.220 km con servicio, 775 km
vacíos y peso 7.665. Este resultado CPU no se inyecta en la salida IQM.

## Qué comparar

Compruebe primero que `dataset_sha256` coincide y que `backend_used` identifica
el backend realmente empleado. Después compare `full_coverage`,
`valid_no_overlap`, `coverage_percent`, `total_empty_km`, `total_weight` y
`number_of_trains`. El resultado también indica servicios descubiertos o
duplicados y kilómetros con servicio.

La CPU informa `solver_time_seconds` para la optimización, además de
`execution_time_seconds` con validación. IQM separa el tiempo de transpilación,
la duración de espera y ejecución del job y el posprocesado. Esas mediciones
**no son directamente equivalentes**: un benchmark temporal necesita fijar
hardware, colas, repeticiones y el mismo perímetro de medida.

## Estado de validación

- La ruta CPU se ha probado con los tres archivos Python aislados y devuelve
  C01 + C03 con cobertura completa.
- La ruta del orquestador a IQM y la evaluación de una muestra conocida se han
  comprobado sin depender de un dispositivo cuántico.
- La instalación de dependencias, la transpilación para el backend elegido y
  la ejecución real en IQM siguen pendientes de prueba en Q-Centroid.

Referencia de contexto: [caso I
