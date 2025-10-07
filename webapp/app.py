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

        # Prepare response: convert DataFrames to HTML
        def df_to_html(df):
            if df is None:
                return ""
            return df.to_html(classes="table table-sm table-bordered", index=False, escape=False)

        tabla_desag = df_to_html(result['disagg']['df'])
        tabla_prod  = df_to_html(result['tabla_produccion'])
        tabla_inv   = df_to_html(result['tabla_inventario'])
        df_totales  = df_to_html(result['sim_totales'])
        df_productos= df_to_html(result['sim_productos'])
        df_estaciones= df_to_html(result['sim_estaciones'])

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
            "figures": figs_base64
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
