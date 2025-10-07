"""
FastAPI backend for the production planning and simulation web application.

This module defines routes for serving the main page and processing
simulation requests. It relies on the `run_full_process` function from
full_process.py to perform the heavy computations. DataFrames are
converted to HTML strings for easy insertion into the page and charts are
rendered as PNG images encoded in base64.
"""

import base64
import io
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from full_process import run_full_process

# Create FastAPI instance
app = FastAPI()

# Mount static directory for any CSS/JS assets if needed
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="webapp/templates")

# Default data and process times to populate the form when the page loads
DEFAULT_DATA = {
    "January":  [75648, 44172, 66960, 31320, 75600],
    "February": [81952, 47853, 72540, 33930, 81900],
    "March":    [98342, 57424, 87048, 40716, 98280],
    "April":    [107168, 62577, 94860, 44370, 107100],
    "May":      [100864, 58896, 89280, 41760, 100800],
    "June":     [103386, 60368, 91512, 42804, 103320],
    "July":     [105907, 61841, 93744, 43848, 105840],
    "August":   [108429, 63313, 95976, 44892, 108360],
    "September":[103386, 60368, 91512, 42804, 103320],
    "October":  [114733, 66994, 101556, 47502, 114660],
    "November": [119776, 69939, 106020, 49590, 119700],
    "December": [141210, 82454, 124992, 58464, 141120]
}
DEFAULT_TIEMPOS = {
    "min_firstS":15, "max_firstS":25,
    "min_pas1":26, "max_pas1":40,
    "min_pas2":18, "max_pas2":28,
    "min_pas3":18, "max_pas3":26,
    "min_pas4":22, "max_pas4":33,
    "min_pas5":24, "max_pas5":38,
    "min_fill":10, "max_fill":15,
    "min_label":18, "max_label":22
}

# Orden de meses para que siempre se rendericen igual
MONTH_ORDER = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """Serve the main page with default data embedded."""
    # Pasamos directamente los diccionarios
    return templates.TemplateResponse("index.html", {
        "request": request,
        "default_data": DEFAULT_DATA,
        "default_times": DEFAULT_TIEMPOS,
        "month_order": MONTH_ORDER
    })


@app.post("/run", response_class=JSONResponse)
async def run_simulation(payload: dict = Body(...)):
    """
    Run the production planning and simulation with the provided parameters.
    """
    try:
        # Extract parameters and apply defaults
        data = payload.get("data") or DEFAULT_DATA
        p_inv_inicial = float(payload.get("p_inv_inicial", 0.25))
        p_inv_final   = float(payload.get("p_inv_final", 0.10))
        tiempos = payload.get("tiempos_procesos") or DEFAULT_TIEMPOS
        tiempos = {k: float(v) for k, v in tiempos.items()}
        unidad = float(payload.get("unidad", 3))
        use_no_consecutive = bool(payload.get("use_no_consecutive", False))
        use_smooth = bool(payload.get("use_smooth", False))
        ct  = float(payload.get("ct", 578))
        ht  = float(payload.get("ht", 145))
        pit = float(payload.get("pit", 10000000))
        crt = float(payload.get("crt", 5931.25))
        cot = float(payload.get("cot", 5931.25))
        cwt = float(payload.get("cwt", 5931.25))
        cwt_prima = float(payload.get("cwt_prima", 5931.25))
        graficar = bool(payload.get("graficar", True))
        costo_prod = float(payload.get("costo_prod", 1.0))
        costo_inv  = float(payload.get("costo_inv", 0.25))
        return_tables = bool(payload.get("return_tables", True))
        make_plots    = bool(payload.get("make_plots", True))
        reps = int(payload.get("reps", 10))
        verbose = bool(payload.get("verbose", True))

        # Coerce month keys to strings just in case
        data = {str(k): [float(x) for x in v] for k, v in data.items()}

        # Run the process
        result = run_full_process(
            data=data,
            p_inv_inicial=p_inv_inicial,
            p_inv_final=p_inv_final,
            tiempos_procesos=tiempos,
            unidad=unidad,
            use_no_consecutive=use_no_consecutive,
            use_smooth=use_smooth,
            ct=ct,
            ht=ht,
            pit=pit,
            crt=crt,
            cot=cot,
            cwt=cwt,
            cwt_prima=cwt_prima,
            graficar=graficar,
            costo_prod=costo_prod,
            costo_inv=costo_inv,
            return_tables=return_tables,
            make_plots=make_plots,
            reps=reps,
            verbose=verbose
        )

        # Prepare response: convert DataFrames to HTML and JSON
        def df_to_html(df, extra_classes: str = ""):
            if df is None:
                return ""
            classes = "table table-sm table-bordered"
            if extra_classes:
                classes = f"{classes} {extra_classes}"
            return df.to_html(classes=classes, index=False, escape=False)

        def df_to_json(df, *, reset_index: bool = True, rename_map: dict | None = None):
            if df is None:
                return None
            df_copy = df.reset_index() if reset_index else df.copy()
            if rename_map:
                df_copy = df_copy.rename(columns=rename_map)
            return df_copy.to_dict(orient='records')

        desagg_df = result['disagg']['df']
        tabla_prod_df = result['tabla_produccion']
        if tabla_prod_df is not None:
            tabla_prod_df = tabla_prod_df.copy().reset_index().rename(columns={"index": "Mes"})
        tabla_inv_df = result['tabla_inventario']
        if tabla_inv_df is not None:
            tabla_inv_df = tabla_inv_df.copy().reset_index().rename(columns={"index": "Mes"})
        sim_totales_df = result['sim_totales']
        sim_productos_df = result['sim_productos']
        sim_estaciones_df = result['sim_estaciones']

        tabla_desag = df_to_html(desagg_df)
        tabla_prod  = df_to_html(tabla_prod_df) if tabla_prod_df is not None else ""
        tabla_inv   = df_to_html(tabla_inv_df) if tabla_inv_df is not None else ""
        df_totales  = df_to_html(sim_totales_df)
        df_productos= df_to_html(sim_productos_df)
        df_estaciones= df_to_html(sim_estaciones_df)

        # JSON serializable tables for frontend interactive plotting
        json_agg_table = df_to_json(result['agg']['df'])
        json_disagg_table = df_to_json(desagg_df)
        json_tabla_produccion = df_to_json(tabla_prod_df, reset_index=False)
        json_tabla_inventario = df_to_json(tabla_inv_df, reset_index=False)
        json_sim_totales = None
        if sim_totales_df is not None:
            rename_totales = dict(zip(sim_totales_df.columns, ["Metrica", "Media", "IC95_HW"]))
            json_sim_totales = df_to_json(sim_totales_df.copy(), reset_index=False, rename_map=rename_totales)
        json_sim_productos = df_to_json(sim_productos_df, reset_index=False)
        json_sim_estaciones = None
        if sim_estaciones_df is not None:
            rename_estaciones = dict(zip(
                sim_estaciones_df.columns,
                ["Estacion", "Capacidad", "Utilizacion_media", "Utilizacion_HW", "Espera_media_min", "Espera_HW"]
            ))
            json_sim_estaciones = df_to_json(sim_estaciones_df.copy(), reset_index=False, rename_map=rename_estaciones)

        # Convert figures (if present) to base64
        figs_base64 = {}
        if make_plots and result.get('figs'):
            for name, fig in result['figs'].items():
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                figs_base64[name] = base64.b64encode(buf.read()).decode('utf-8')
                fig.clf()

        # Build JSON response
        return JSONResponse({
            "status": result['agg']['status'],
            "z": result['agg']['z'],
            "tabla_desag": tabla_desag,
            "tabla_prod": tabla_prod,
            "tabla_inv": tabla_inv,
            "df_totales": df_totales,
            "df_productos": df_productos,
            "df_estaciones": df_estaciones,
            "figures": figs_base64,
            # JSON data for interactive plots
            "json_agg_table": json_agg_table,
            "json_disagg_table": json_disagg_table,
            "json_tabla_produccion": json_tabla_produccion,
            "json_tabla_inventario": json_tabla_inventario,
            "json_sim_totales": json_sim_totales,
            "json_sim_productos": json_sim_productos,
            "json_sim_estaciones": json_sim_estaciones
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
