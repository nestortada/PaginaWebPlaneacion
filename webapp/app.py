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
import math
import random
import traceback
from functools import lru_cache
from numbers import Number
from pathlib import Path
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Create FastAPI instance
app = FastAPI()

# Resolve paths from this file to avoid cwd-dependent issues in serverless runtimes.
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Mount static directory for any CSS/JS assets if needed.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

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


def generate_random_default_data(seed: int | None = None):
    """Generate a plausible monthly demand matrix with mixed up/down behavior."""
    rng = random.Random(seed)
    base_template = [86000, 52000, 76000, 37000, 86000]
    # Keep values closer to a centered baseline (less dispersion across months/products)
    base_levels = [int(base * rng.uniform(0.96, 1.12)) for base in base_template]
    season_amplitudes = [rng.uniform(0.025, 0.08) for _ in range(5)]
    phases = [rng.uniform(0, 2 * math.pi) for _ in range(5)]

    data = {}
    max_idx = max(1, len(MONTH_ORDER) - 1)

    # Enforce mixed directions at the beginning and at the end of the horizon.
    # +1 means uptrend, -1 means downtrend.
    start_signs = [1, -1, 1, 1, -1]
    end_signs =   [-1, 1, -1, -1, 1]
    # Tea (index 3) specifically: starts up, later turns down.
    start_signs[3] = 1
    end_signs[3] = -1

    start_slopes = [sign * rng.uniform(0.05, 0.12) for sign in start_signs]
    end_slopes = [sign * rng.uniform(0.05, 0.12) for sign in end_signs]
    # Relative competition: some products gain demand share and overtake others over time.
    relative_shift = [rng.uniform(-0.05, 0.05) for _ in range(5)]
    winners = rng.sample(range(5), k=2)
    losers = rng.sample([idx for idx in range(5) if idx not in winners], k=2)
    for idx in winners:
        relative_shift[idx] = rng.uniform(0.12, 0.22)
    for idx in losers:
        relative_shift[idx] = -rng.uniform(0.12, 0.22)

    # Independent product behavior: no global month multiplier shared by all products.
    prod_momentum = [rng.uniform(-0.025, 0.025) for _ in range(5)]
    prod_ar = [rng.uniform(0.40, 0.72) for _ in range(5)]
    shock_prob = [rng.uniform(0.12, 0.28) for _ in range(5)]
    for month_idx, month in enumerate(MONTH_ORDER):
        row = []
        for prod_idx in range(5):
            t = month_idx / max_idx
            # Quadratic trend with controlled initial and final slope:
            # trend'(0)=start_slope and trend'(1)=end_slope
            s0 = start_slopes[prod_idx]
            s1 = end_slopes[prod_idx]
            trend = 1.0 + (s0 * t) + (0.5 * (s1 - s0) * (t ** 2))
            season = 1.0 + season_amplitudes[prod_idx] * math.sin((2 * math.pi * month_idx / 12) + phases[prod_idx])
            competition_factor = 1.0 + (relative_shift[prod_idx] * ((2.0 * t) - 1.0))
            prod_momentum[prod_idx] = (prod_ar[prod_idx] * prod_momentum[prod_idx]) + rng.uniform(-0.018, 0.018)
            momentum_factor = 1.0 + prod_momentum[prod_idx]
            noise = 1.0 + rng.uniform(-0.025, 0.025)
            if rng.random() < shock_prob[prod_idx]:
                noise *= rng.uniform(0.94, 1.08)
            value = base_levels[prod_idx] * trend * season * competition_factor * momentum_factor * noise

            # Clamp to keep demand centered and avoid unrealistic dispersion.
            min_allowed = base_levels[prod_idx] * 0.76
            max_allowed = base_levels[prod_idx] * 1.30
            value = min(max(value, min_allowed), max_allowed)

            row.append(max(10000, int(round(value))))
        data[month] = row
    return data

def round_dataframe(df, decimals: int = 2):
    """Return a rounded copy of the DataFrame for consistent presentation."""
    if df is None:
        return None
    df_copy = df.copy()
    if hasattr(df_copy, "select_dtypes"):
        numeric_cols = df_copy.select_dtypes(include="number").columns
        if len(numeric_cols):
            df_copy[numeric_cols] = df_copy[numeric_cols].round(decimals)
    return df_copy


def round_value(value, decimals: int = 2):
    """Round numeric scalars, leave other values untouched."""
    return round(float(value), decimals) if isinstance(value, Number) else value


@lru_cache(maxsize=1)
def get_run_full_process():
    """Lazy-load heavy optimization module to reduce cold-start failures."""
    from full_process import run_full_process
    return run_full_process

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """Serve the main page with default data embedded."""
    # Random but plausible demand defaults on each page load.
    random_default_data = generate_random_default_data()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "default_data": random_default_data,
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
        use_safe_stock = bool(payload.get("use_safe_stock", False))
        use_smooth = bool(payload.get("use_smooth", False))
        alpha = float(payload.get("alpha", 0.5))
        smooth_pct = float(payload.get("smooth_pct", 0.20))
        ct  = float(payload.get("ct", 578))
        ht  = float(payload.get("ht", 145))
        pit = float(payload.get("pit", 10000000))
        crt = float(payload.get("crt", 5931.25))
        cot = float(payload.get("cot", 5931.25))
        cwt = float(payload.get("cwt", 5931.25))
        cwt_prima = float(payload.get("cwt_prima", 5931.25))
        costo_maquina = float(payload.get("costo_maquina", 10000))
        graficar = bool(payload.get("graficar", True))
        costo_prod = float(payload.get("costo_prod", 1.0))
        costo_inv  = float(payload.get("costo_inv", 0.25))
        return_tables = bool(payload.get("return_tables", True))
        make_plots    = bool(payload.get("make_plots", False))
        reps = int(payload.get("reps", 10))
        verbose = bool(payload.get("verbose", True))
        capacidades_override = payload.get("capacidades_override")
        if not isinstance(capacidades_override, list):
            capacidades_override = None

        # Coerce month keys to strings just in case
        data = {str(k): [float(x) for x in v] for k, v in data.items()}

        # Run the process
        run_full_process = get_run_full_process()
        result = run_full_process(
            data=data,
            p_inv_inicial=p_inv_inicial,
            p_inv_final=p_inv_final,
            tiempos_procesos=tiempos,
            unidad=unidad,
            use_no_consecutive=use_no_consecutive,
            use_safe_stock=use_safe_stock,
            alpha=alpha,
            use_smooth=use_smooth,
            smooth_pct=smooth_pct,
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
            capacidades_override=capacidades_override,
            reps=reps,
            verbose=verbose
        )

        # Prepare response: convert DataFrames to HTML and JSON
        def df_to_html(df, extra_classes: str = ""):
            if df is None:
                return ""
            classes = "w-full text-sm text-left text-gray-300"
            if extra_classes:
                classes = f"{classes} {extra_classes}"
            df_fmt = round_dataframe(df)
            
            # Use pandas Styler to add tailwind classes to th and td, or just return basic html and let CSS handle it.
            # Easiest is to generate raw HTML and let index.html's CSS target the table elements, 
            # or apply basic Tailwind classes directly to the table tag.
            html = df_fmt.to_html(
                classes=classes,
                index=False,
                escape=False,
                float_format=lambda x: f"{x:.2f}"
            )
            # Remove any pandas default styling that might override ours
            html = html.replace(' style="text-align: right;"', '')
            
            # Add basic Tailwind styling and ensure left alignment
            html = html.replace('<th>', '<th class="px-4 py-3 bg-white/5 border-b border-white/10 font-semibold text-white text-left">')
            html = html.replace('<td>', '<td class="px-4 py-3 border-b border-white/10 text-left">')
            html = html.replace('<tr>', '<tr class="hover:bg-white/5 transition-colors">')
            
            return html

        def df_to_json(df, *, reset_index: bool = True, rename_map: dict | None = None, decimals: int = 2):
            if df is None:
                return None
            df_copy = df.reset_index() if reset_index else df.copy()
            if rename_map:
                df_copy = df_copy.rename(columns=rename_map)
            df_copy = round_dataframe(df_copy, decimals=decimals)
            return df_copy.to_dict(orient='records')

        desagg_df = result['disagg']['df'].copy()
        if "Producto_lbl" in desagg_df.columns:
            desagg_df = desagg_df.drop(columns=["Producto_lbl"])
        if "Producto" in desagg_df.columns: # Also remove numeric ID if present
            desagg_df = desagg_df.drop(columns=["Producto"])

        tabla_prod_df = result['tabla_produccion']
        if tabla_prod_df is not None:
            tabla_prod_df = tabla_prod_df.copy()
            if "Producto_lbl" in tabla_prod_df.columns:
                tabla_prod_df = tabla_prod_df.drop(columns=["Producto_lbl"])
            tabla_prod_df = tabla_prod_df.reset_index().rename(columns={"index": "Mes"})

        tabla_inv_df = result['tabla_inventario']
        if tabla_inv_df is not None:
            tabla_inv_df = tabla_inv_df.copy()
            if "Producto_lbl" in tabla_inv_df.columns:
                tabla_inv_df = tabla_inv_df.drop(columns=["Producto_lbl"])
            tabla_inv_df = tabla_inv_df.reset_index().rename(columns={"index": "Mes"})

        sim_totales_df = result['sim_totales']
        sim_productos_df = result['sim_productos']
        sim_estaciones_df = result['sim_estaciones']
        capacities_info = result.get('capacities') or {}
        agg_status = result.get('agg', {}).get('status')
        agg_z = round_value(result.get('agg', {}).get('z'))
        used_capacities = capacities_info.get("used") if isinstance(capacities_info, dict) else None
        total_maquinas = 0.0
        if isinstance(used_capacities, list):
            for cap in used_capacities:
                try:
                    total_maquinas += max(0.0, float(cap))
                except (TypeError, ValueError):
                    continue
        costo_maquinas_total = round_value(total_maquinas * costo_maquina)
        agg_z_total = None
        if isinstance(agg_z, Number):
            agg_z_total = round_value(float(agg_z) + (total_maquinas * costo_maquina))

        tabla_desag = df_to_html(desagg_df)
        tabla_prod  = df_to_html(tabla_prod_df) if tabla_prod_df is not None else ""
        tabla_inv   = df_to_html(tabla_inv_df) if tabla_inv_df is not None else ""
        df_totales  = df_to_html(sim_totales_df)
        df_productos= df_to_html(sim_productos_df)
        df_estaciones= df_to_html(sim_estaciones_df)

        # Calculate totals for KPI cards
        total_demand = sum(sum(v) for v in data.values())
        total_production = tabla_prod_df.iloc[:, 1:].sum().sum() if tabla_prod_df is not None else 0

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
                ["Estacion", "Capacidad_ideal", "Capacidad", "Utilizacion_media", "Utilizacion_HW", "Espera_media_min", "Espera_HW"]
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
            "status": agg_status,
            "z_base": agg_z,
            "z": agg_z,
            "z_total": agg_z_total,
            "costo_maquina": round_value(costo_maquina),
            "total_maquinas": round_value(total_maquinas),
            "costo_maquinas_total": costo_maquinas_total,
            "total_demand": total_demand,
            "total_production": total_production,
            "tabla_desag": tabla_desag,
            "tabla_prod": tabla_prod,
            "tabla_inv": tabla_inv,
            "df_totales": df_totales,
            "df_productos": df_productos,
            "df_estaciones": df_estaciones,
            "capacities": capacities_info,
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
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc(limit=8)},
            status_code=500
        )
