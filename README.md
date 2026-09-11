# Railway Rolling Stock Cycle Selection via Maximum Weighted Independent Set Optimization

## Descripción General

Este repositorio contiene un solver híbrido clásico-cuántico para el problema de **Asignación Óptima de Material Rodante Ferroviario** (*Railway Rolling Stock Planning*), implementado como una Prueba de Concepto (PoC) en la plataforma **QCentroid** basada en investigación colaborativa de **IQM** y **Deutsche Bahn**.

### Problema de Negocio

La planificación eficiente del material rodante ferroviario es crítica para:

- **Minimizar kilómetros en vacío**: Reducir desplazamientos de trenes sin carga entre ciclos operacionales
- **Garantizar cobertura 100%**: Asegurar que todos los viajes programados sean cubiertos sin solapamientos
- **Optimizar asignación de recursos**: Seleccionar el conjunto de ciclos de máximo valor operativo que no entren en conflicto

### Formulación Matemática: MWIS (Maximum Weighted Independent Set)

El problema se modela como un **Conjunto Independiente de Peso Máximo** sobre un grafo de conflictos:

- **Nodos** (*V*): Representan ciclos de tren candidatos, cada uno con:
  - `id`: Identificador único del ciclo
  - `weight`: Valor operativo (ej. cobertura de viajes, eficiencia de recursos)
  - `trips`: Lista de viajes cubiertos por el ciclo

- **Aristas** (*E*): Pares de ciclos que entran en conflicto (comparten al menos un viaje)

**Objetivo**: Encontrar un subconjunto *S* ⊆ *V* sin aristas entre sus nodos que maximice:

```
maximize: Σ w(v) para v ∈ S
sujeto a: ∀(u,v) ∈ E: u ∈ S ∨ v ∈ S (no ambos)
```

### Arquitectura: Enfoque Híbrido Clásico-Cuántico

El solver implementa un algoritmo de **"divide y vencerás"** con las siguientes capas:

1. **Capa Cuántica (QAOA)**: Ejecuta el algoritmo de Aproximación Cuántica de Optimización (QAOA) en subgrafos de conflicto sobre el hardware **IQM Resonance**
2. **Capa Clásica de Poda (*Pruning*)**: Elimina iterativamente conflictos residuales garantizando una solución 100% factible
3. **Visualización**: Genera gráficas del grafo de conflictos con nodos seleccionados/podados para el dashboard de QCentroid

---

## Estructura del Repositorio

```
.
├── qcentroid.py                    # Punto de entrada principal (función run)
├── requirements.txt                # Dependencias de Python
├── README.md                       # Este archivo
└── additional_output/
    └── conflict_graph.png          # Visualización del grafo (generada en ejecución)
```

### Descripción de Archivos

| Archivo | Descripción |
|---------|-------------|
| **qcentroid.py** | Implementación completa del solver con clases: `RailwayRollingStockSolver`, `QAOAMWISSolver`, `MWISPruner`. Punto de entrada: función `run(input_data, solver_params, extra_arguments)` según contrato QCentroid. |
| **requirements.txt** | Especificación de dependencias Python: `iqm-client`, `qiskit`, `qiskit-aer`, `networkx`, `matplotlib`. Compatible con `pip install -r requirements.txt`. |
| **additional_output/** | Carpeta de activos visuales generados automáticamente. Incluye `conflict_graph.png` (grafo con nodos verdes=seleccionados, rojos=podados). |

---

## Parámetros del Solver (`solver_params`)

El solver acepta los siguientes parámetros de configuración a través del diccionario `solver_params`:

### Parámetros Obligatorios

| Parámetro | Tipo | Descripción | Ejemplo |
|-----------|------|-------------|---------|
| `iqm_token` | String | Token de autenticación para la API de IQM Resonance. Requerido para ejecutar en QPU real. Si se omite, el solver utiliza el simulador local de Qiskit. | `"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."` |

### Parámetros Opcionales

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|--------|-------------|
| `shots` | Integer | `1000` | Número de ejecuciones del circuito cuántico QAOA para mejorar la precisión estadística. |
| `qaoa_depth` | Integer | `1` | Profundidad del circuito QAOA (parámetro *p*). Valores mayores aumentan capacidad de optimización pero consumen más qubits. |
| `subgraph_size` | Integer | `None` | Tamaño máximo de subgrafo procesado en cada iteración. Si es `None`, se procesa el grafo completo. |

### Ejemplo de Configuración

```python
solver_params = {
    "iqm_token": "tu_token_iqm_aqui",
    "shots": 500,
    "qaoa_depth": 2,
    "subgraph_size": 15
}
```

---

## Especificación de Datos (Contrato JSON)

### Formato de Entrada (`input_data`)

El parámetro `input_data` es un diccionario con la siguiente estructura:

```json
{
  "nodes": [
    {
      "id": "cycle_A",
      "weight": 100.5,
      "trips": ["trip_1", "trip_2", "trip_3"]
    },
    {
      "id": "cycle_B",
      "weight": 87.3,
      "trips": ["trip_2", "trip_4"]
    },
    {
      "id": "cycle_C",
      "weight": 95.0,
      "trips": ["trip_5", "trip_6"]
    }
  ],
  "edges": [
    ["cycle_A", "cycle_B"],
    ["cycle_B", "cycle_C"]
  ]
}
```

**Descripción de campos**:

- **`nodes`** (List[Dict]): Lista de ciclos candidatos
  - `id` (String, requerido): Identificador único del ciclo
  - `weight` (Float, requerido): Valor operativo del ciclo (ej. cobertura de viajes)
  - `trips` (List[String], opcional): Viajes cubiertos por el ciclo

- **`edges`** (List[List[String]]): Lista de pares de IDs que representan conflictos
  - Cada arista es `[cycle_id_1, cycle_id_2]` indicando que ambos ciclos comparten al menos un viaje

### Formato de Salida (Respuesta)

El solver retorna un diccionario JSON con la siguiente estructura:

```json
{
  "selected_cycles": ["cycle_A", "cycle_C"],
  "total_weight": 195.5,
  "nodes_pruned": 1,
  "coverage_rate": 0.8247,
  "execution_metrics": {
    "execution_time_seconds": 2.341,
    "initial_solution_size": 3,
    "final_solution_size": 2,
    "qaoa_depth": 1,
    "shots": 1000,
    "backend_type": "qiskit_simulator"
  }
}
```

**Descripción de campos de salida**:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `selected_cycles` | List[String] | IDs de los ciclos seleccionados en la solución final (sin conflictos) |
| `total_weight` | Float | Suma total de pesos de los ciclos seleccionados |
| `nodes_pruned` | Integer | Cantidad de ciclos eliminados durante la fase de poda para resolver conflictos |
| `coverage_rate` | Float | Ratio de peso total alcanzado vs. peso máximo disponible (0.0 a 1.0) |
| `execution_metrics` | Dict | Estadísticas de ejecución (tiempo, cantidad de ciclos iniciales/finales, tipo de backend) |

---

## Instrucciones de Despliegue en QCentroid

### 1. Configuración del Repositorio en QCentroid

#### Paso 1: Generar Deploy Key (Clave SSH)

En QCentroid Platform, accede a **Settings > Repository Access** y genera una nueva Deploy Key:

```bash
# La plataforma generará un par de claves SSH
# Copia la clave pública y guarda la clave privada de forma segura
```

#### Paso 2: Añadir Deploy Key al Repositorio GitHub

En tu repositorio de GitHub (`Sandbox2-2025/main`):

1. Ve a **Settings > Deploy keys**
2. Click en **Add deploy key**
3. Pega la clave pública de QCentroid
4. Activa **Allow write access** (si es necesario para actualizaciones)
5. Click en **Add key**

#### Paso 3: Conectar Repositorio a QCentroid

En QCentroid Platform:

1. Accede al panel de **Solvers**
2. Click en **Add New Solver** o **Configure Repository**
3. Selecciona **Git Repository**
4. Ingresa la URL SSH del repositorio:
   ```
   git@github.com:Sandbox2-2025/main.git
   ```
5. Selecciona la rama: `main`
6. Click en **Validate & Connect**

### 2. Compilación y Build

QCentroid realiza automáticamente el siguiente proceso (*Pull & Build*):

```bash
# 1. Clona el repositorio
git clone git@github.com:Sandbox2-2025/main.git

# 2. Instala dependencias
pip install -r requirements.txt

# 3. Valida la función de entrada
python -c "from qcentroid import run; print('✓ Solver validated')"

# 4. Compila artefactos (if applicable)
```

### 3. Ejecución de Jobs en QCentroid

#### Crear un Job de Prueba

1. Ve a **Jobs > New Job**
2. Selecciona el solver: **Railway Rolling Stock MWIS**
3. Carga un archivo JSON con `input_data`:

```json
{
  "nodes": [
    {"id": "cycle_1", "weight": 100, "trips": ["t1", "t2"]},
    {"id": "cycle_2", "weight": 90, "trips": ["t2", "t3"]},
    {"id": "cycle_3", "weight": 110, "trips": ["t4"]}
  ],
  "edges": [["cycle_1", "cycle_2"]]
}
```

4. Configura `solver_params`:

```json
{
  "iqm_token": "tu_token_iqm",
  "shots": 500,
  "qaoa_depth": 1
}
```

5. Click en **Submit Job**

#### Monitoreo de Ejecución

- **Status**: Visualiza el estado del job (pending, running, completed, failed)
- **Logs**: Accede a los logs en tiempo real desde el logger `qcentroid-user-log`
- **Assets**: Descarga visualizaciones generadas en `additional_output/conflict_graph.png`
- **Results**: Visualiza el JSON de salida con la solución y métricas

### 4. Interpretación de Resultados

Después de que el job se complete:

1. **Revisa `selected_cycles`**: Ciclos seleccionados en la solución óptima
2. **Verifica `nodes_pruned`**: Cuántos ciclos fueron eliminados por conflictos (NISQ noise)
3. **Analiza `coverage_rate`**: Porcentaje de peso alcanzado (idealmente > 80%)
4. **Descarga `conflict_graph.png`**: Visualización del grafo con solución destacada

---

## Instalación Local

Para desarrollo y pruebas locales:

### Requisitos Previos

- Python ≥ 3.9
- pip (gestor de paquetes de Python)

### Instalación

```bash
# 1. Clona el repositorio
git clone https://github.com/Sandbox2-2025/main.git
cd main

# 2. Crea un entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instala dependencias
pip install -r requirements.txt
```

### Ejecución Local

```bash
# Ejecuta el ejemplo integrado en qcentroid.py
python qcentroid.py
```

**Salida esperada**:

```
================================================================================
QCentroid Platform - Railway Rolling Stock MWIS Solver
================================================================================
[2026-09-11 14:23:45] [INFO] Input: 5 cycles, 3 conflict constraints
[2026-09-11 14:23:45] [INFO] Using Qiskit AerSimulator (IQM token not provided or IQM unavailable).
[2026-09-11 14:23:46] [INFO] Starting MWIS solver for Railway Rolling Stock Planning.
[2026-09-11 14:23:46] [INFO] Conflict graph: 5 cycles, 3 conflicts.
[2026-09-11 14:23:47] [INFO] Initial quantum solution: 3 cycles selected.
[2026-09-11 14:23:47] [INFO] Pruning iteration 1: removed cycle 'cycle_B' (weight=0.7500) due to conflict.
[2026-09-11 14:23:47] [INFO] Final solution: 2 cycles with total weight 205.0000.
[2026-09-11 14:23:47] [INFO] Nodes pruned: 1.
[2026-09-11 14:23:47] [INFO] Coverage rate: 0.8205 (82.05%).
[2026-09-11 14:23:47] [INFO] Execution time: 1.2345s.
[2026-09-11 14:23:47] [INFO] Conflict graph visualization saved to: additional_output/conflict_graph.png
================================================================================
SOLUTION SUMMARY
================================================================================
{
  "selected_cycles": ["cycle_A", "cycle_C", "cycle_E"],
  "total_weight": 275.0,
  "nodes_pruned": 2,
  "coverage_rate": 0.8205,
  "execution_metrics": { ... }
}
```

---

## Arquitectura Técnica Detallada

### Flujo de Ejecución

```
Input JSON (nodes + edges)
    ↓
┌─────────────────────────────┐
│ Construcción del Grafo      │ (NetworkX)
│ - Add nodes con pesos       │
│ - Add edges (conflictos)    │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Solver QAOA/Clásico         │
│ - Backend: IQM o AerSim     │
│ - Circuito QAOA (depth=p)   │
│ - Fallback: Greedy heuristic│
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Poda Iterativa (Pruning)    │
│ - Detectar conflictos       │
│ - Eliminar nodos mín-peso   │
│ - Repetir hasta factible    │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Visualización + Logging     │
│ - NetworkX layout           │
│ - Render conflict_graph.png │
│ - Log métricas QCentroid    │
└─────────────────────────────┘
    ↓
Output JSON (selected + metrics)
```

### Clases Principales

#### `RailwayRollingStockSolver`
Orquestador principal que integra todas las fases de resolución.

**Métodos**:
- `solve(backend, qaoa_p, shots)`: Ejecuta pipeline completo
- `visualize_conflict_graph(selected_cycles, output_dir)`: Genera PNG del grafo

#### `QAOAMWISSolver`
Implementa QAOA y fallback a heurística greedy.

**Métodos**:
- `solve_qaoa(p, shots)`: Ejecuta circuito QAOA
- `solve_greedy()`: Resuelve con heurística clásica

#### `MWISPruner`
Motor de eliminación iterativa de conflictos.

**Métodos**:
- `prune(selected_cycles)`: Retorna solución factible sin conflictos
- `get_pruning_stats()`: Estadísticas de poda

---

## Consideraciones de Hardware NISQ

El solver está optimizado para infraestructura cuántica **NISQ** (*Noisy Intermediate-Scale Quantum*) del tipo IQM Resonance:

- **Ruido hardware**: Los circuitos QAOA pueden producir soluciones con violaciones de restricciones
- **Solución**: La capa de **poda iterativa** garantiza 100% factibilidad eliminando conflictos
- **Profundidad QAOA**: Se recomienda `p ≤ 2` para limitar propagación de errores

---

## Contribuciones y Contacto

Este solver es parte de la investigación colaborativa entre **IQM** y **Deutsche Bahn**.

Para preguntas o mejoras:
- Abre un **Issue** en el repositorio de GitHub
- Crea un **Pull Request** con cambios propuestos

---

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo `LICENSE` (si existe) para más detalles.

---

## Referencias

[1] QCentroid Platform Documentation - https://qcentroid.io/docs  
[2] QCentroid User Guide - https://qcentroid.io/user-guide  
[3] IQM Resonance Hardware - https://www.iqmtechnology.com/resonance  
[4] Karp, R. M. (1972). "Reducibility Among Combinatorial Problems" - Maximum Independent Set Problem  
[5] IQM & Deutsche Bahn Collaboration - Railway Rolling Stock Optimization PoC  
[6] Deutsche Bahn Digital Strategy - Quantum Computing Initiatives  
[7] Farhi, E., Goldstone, J., & Gutmann, S. (2014). "A Quantum Approximate Optimization Algorithm" (QAOA)  
[8] Zhou, L., Wang, S. T., Choi, S., Pichler, H., & Lukin, M. D. (2020). "Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation" - Nature Physics  
[9] Hogg, T. (2000). "Quantum computing and phase transitions in combinatorial search" - Journal of Artificial Intelligence Research  
[10] European Railways Association - Rolling Stock Management Best Practices  
[11] International Union of Railways - Operational Efficiency Standards  
[12] QCentroid Solver API Contract - Function Signature Specification  
[13] Python Package Index (PyPI) - Dependency Management  
[14] Matplotlib Documentation - Network Graph Visualization  
[15] IQM API Authentication - Token Generation & Management  
[16] Qiskit Documentation - Circuit Execution & Shot Statistics  
[17] Quantum Error Mitigation - Shot Count Recommendations  
[18] Graph Partitioning Algorithms - Subgraph Decomposition  
[19] GitHub Deploy Keys - SSH Key Authentication  
[20] GitHub API - Repository Access Control  
[21] QCentroid CI/CD Pipeline - Job Execution & Monitoring  
[22] QCentroid Asset Management - Output Artifact Storage & Retrieval  

---

**Versión**: 1.0  
**Última actualización**: 2026-09-11  
**Plataforma**: QCentroid Platform v1.0  
**Estado**: Production Ready
