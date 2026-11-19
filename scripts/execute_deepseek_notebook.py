"""Execute offline notebook with the invoking Python interpreter."""
import asyncio
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root=Path(__file__).resolve().parents[1]
if sys.platform=='win32':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
for key,folder in [('JUPYTER_RUNTIME_DIR','jupyter_runtime'),('IPYTHONDIR','ipython'),('JUPYTER_CONFIG_DIR','jupyter_config'),('JUPYTER_DATA_DIR','jupyter_data')]:
    directory=root/'.cache'/folder;directory.mkdir(parents=True,exist_ok=True);os.environ[key]=str(directory)
path=root/'notebooks/deepseek_analysis.ipynb'
notebook=nbformat.read(path,as_version=4)
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(notebook,timeout=300,km=manager,resources={'metadata':{'path':str(root)}}).execute()
nbformat.validate(notebook);nbformat.write(notebook,path)
print('Executed',sum(c.cell_type=='code' for c in notebook.cells),'code cells; no generations')
