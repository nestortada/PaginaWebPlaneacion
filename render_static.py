"""
Render the Jinja2 template with default data into a static HTML file under docs/ and copy static assets.

Usage:
  python render_static.py

This produces a `docs/` folder suitable for publishing with GitHub Pages (branch: master, source: /docs).
"""
import json
import os
import shutil
import stat
from jinja2 import Environment, FileSystemLoader

ROOT = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(ROOT, 'webapp', 'templates')
STATIC_DIR = os.path.join(ROOT, 'webapp', 'static')
OUT_DIR = os.path.join(ROOT, 'docs')


def load_defaults():
    path = os.path.join(ROOT, 'webapp', 'data', 'defaults.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def render_index(defaults):
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=True)
    tpl = env.get_template('index.html')
    out = tpl.render(request={'path':'/'}, default_data=defaults['default_data'], default_times=defaults['default_times'], month_order=defaults['month_order'])
    return out


def ensure_out():
    # Try to remove the output dir safely, handle Windows permission errors
    if os.path.exists(OUT_DIR):
        def on_rm_error(func, path, exc_info):
            # if permission error, try to change file to writable and retry
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                # If still failing, raise the original exception
                raise

        try:
            shutil.rmtree(OUT_DIR, onerror=on_rm_error)
        except Exception as e:
            # If rmtree failed due to external lock (OneDrive, explorer), provide guidance
            print('Warning: no se pudo eliminar docs/ automáticamente:', e)
            print('Intenta cerrar OneDrive o el Explorador de archivos, o elimina la carpeta docs/ manualmente, luego vuelve a ejecutar.')
            raise
    os.makedirs(OUT_DIR, exist_ok=True)


def copy_static():
    if os.path.exists(STATIC_DIR):
        shutil.copytree(STATIC_DIR, os.path.join(OUT_DIR, 'static'))


def write_nojekyll():
    open(os.path.join(OUT_DIR, '.nojekyll'), 'w', encoding='utf-8').close()


def main():
    defaults = load_defaults()
    html = render_index(defaults)
    ensure_out()
    # write index.html
    with open(os.path.join(OUT_DIR, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    copy_static()
    write_nojekyll()
    print('Static site generated in', OUT_DIR)


if __name__ == '__main__':
    main()
