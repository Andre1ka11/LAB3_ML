"""Run the offline analysis notebook in a local kernel, without API calls."""
import asyncio
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
if sys.platform=='win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
for name,subdir in [('JUPYTER_RUNTIME_DIR','.cache/jupyter_runtime'),('IPYTHONDIR','.cache/ipython'),('JUPYTER_CONFIG_DIR','.cache/jupyter_config'),('JUPYTER_DATA_DIR','.cache/jupyter_data')]:
    directory=root/subdir; directory.mkdir(parents=True,exist_ok=True); os.environ[name]=str(directory)
path=root/'notebooks/yandex_analysis.ipynb'
notebook=nbformat.read(path,as_version=4)
NotebookClient(notebook,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
nbformat.validate(notebook)
nbformat.write(notebook,path)
print('Executed analysis notebook:',sum(c.cell_type=='code' for c in notebook.cells),'code cells; API calls: 0')
