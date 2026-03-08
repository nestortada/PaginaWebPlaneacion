# Planeación de Producción y Simulación

Aplicación web para apoyar decisiones de **planeación agregada y desagregada de producción** en una planta de bebidas, con simulación operativa por estaciones.

## ¿Qué resuelve este proyecto?

Este proyecto ayuda a responder preguntas clave de operación:

- Cuánto producir por mes para cubrir demanda con menor costo.
- Cómo balancear inventario inicial/final y evitar escasez.
- Qué capacidad mensual se requiere y cuántas máquinas se necesitan.
- Cómo se distribuye la producción por producto (desagregación).
- Qué tan cargadas están las estaciones del proceso (utilización, espera, cuellos de botella).

En resumen: convierte una matriz de demanda y parámetros de costo/tiempo en un plan de producción con métricas para decidir.

## Alcance funcional

- Interfaz web para cargar/editar demanda mensual.
- Parámetros de costos, inventarios, suavización y restricciones.
- Modelo de optimización (PuLP; fallback con SciPy si PuLP no está disponible).
- Simulación de estaciones de proceso.
- Tablas de resultados y gráficos para análisis.

## Estructura del proyecto

- `webapp/app.py`: backend FastAPI y endpoints (`/`, `/run`).
- `webapp/templates/index.html`: interfaz principal.
- `full_process.py`: lógica de optimización y simulación.
- `requirements.txt`: dependencias de Python.
- `vercel.json`: configuración de despliegue en Vercel.
- `render_static.py`: genera versión estática en `docs/` (opcional).

## Requisitos

- Python 3.11 recomendado.
- `pip` actualizado.
- Git (opcional, para versionar y desplegar).

## Cómo hacerlo funcionar (paso a paso)

## 1. Clonar y entrar al proyecto

```bash
git clone <URL_DEL_REPO>
cd PaginaWebPlaneacion
```

## 2. Crear y activar entorno virtual

En Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows (PowerShell):

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 4. Ejecutar en local

```bash
uvicorn webapp.app:app --host 0.0.0.0 --port 8000 --reload
```

Abrir en navegador:

`http://localhost:8000`

## 5. Usar la aplicación

1. Ajustar la matriz de demanda por mes/producto.
2. Configurar tiempos de proceso y costos.
3. Activar restricciones opcionales (inventario de seguridad, suavización, etc.).
4. Ejecutar simulación.
5. Revisar:
   - Costo total y costo base.
   - Producción e inventario por mes.
   - Capacidades y máquinas requeridas.
   - Métricas de simulación por estación.

## Despliegue en Vercel

## 1. Subir cambios al repositorio

```bash
git add .
git commit -m "Deploy production planning app"
git push
```

## 2. Crear proyecto en Vercel

- Importar el repositorio.
- Framework: detectar automáticamente (Python).
- Mantener `vercel.json` del proyecto.

## 3. Verificar que cargue

- Abrir la URL de Vercel.
- Validar que `/` renderice la interfaz.
- Ejecutar una simulación para validar `/run`.

## Solución de problemas comunes

- Error 500 `FUNCTION_INVOCATION_FAILED` en Vercel:
  - Revisar logs de Functions en Vercel.
  - Confirmar que `requirements.txt` esté completo.
  - Verificar que `webapp/templates` y `webapp/static` existan.

- El modelo tarda mucho:
  - Reducir `reps` de simulación.
  - Desactivar opciones complejas cuando no sean necesarias.

- Problemas con dependencias:
  - Recrear entorno virtual.
  - Reinstalar con `pip install -r requirements.txt`.

## Notas técnicas

- Endpoint principal:
  - `GET /`: renderiza la UI.
  - `POST /run`: ejecuta optimización + simulación y retorna resultados.

- El backend está orientado a escenarios de planeación mensual de 5 familias de producto.

## Próximas mejoras sugeridas

- Exportar resultados a Excel/PDF.
- Escenarios comparativos A/B de costos y capacidad.
- Validación más estricta de datos de entrada en frontend y backend.
