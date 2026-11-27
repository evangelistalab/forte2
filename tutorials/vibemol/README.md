# Cyclopropene in VibeMol

Install Forte2 and JupyterLab in the notebook kernel environment. Install
vibemol-widget 0.2.0 or later from a checkout of VibeMol main in the same
environment.

```bash
python -m pip install -e /path/to/vibemol/python
jupyter lab tutorials/vibemol/Tutorial_VibeMol.ipynb
```

Choose **Run → Run All Cells**. The viewer opens vibemol.org by default.

For a local viewer, serve the VibeMol checkout in a separate terminal.

```bash
cd /path/to/vibemol
python -m http.server 8000 --bind 127.0.0.1
```

From the Forte2 root, launch JupyterLab with the local URL.

```bash
VIBEMOL_VIEWER_URL=http://127.0.0.1:8000/ jupyter lab tutorials/vibemol/Tutorial_VibeMol.ipynb
```

This tutorial lives in a subfolder so the root-level `pytest --nbmake *.ipynb`
CI command skips it. Its widget sends commands through a live browser, so
running the notebook in a kernel alone cannot produce the snapshots.
