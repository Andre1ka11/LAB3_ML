"""Create a project-only GigaChat CA bundle; never changes OS trust or certifi."""
import hashlib
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
CERTIFICATES={
 'russian_trusted_root_ca_pem.crt':'936a43fea6e8e525bcc0f81acd9c3d21b4fc4b9b68acea7906d698005afc6504',
 'russian_trusted_sub_ca_pem.crt':'f0ae589f36774f29ef3648f7984b08d42fcce6f1ffeeb6236d773daeb2744ea6'}


def prepare():
    directory=ROOT/'.cache/gigachat-ca';directory.mkdir(parents=True,exist_ok=True)
    certs=[]
    for name,expected in CERTIFICATES.items():
        path=directory/name
        if not path.exists():
            response=requests.get('https://gu-st.ru/content/lending/'+name,timeout=60)
            response.raise_for_status();data=response.content
        else:data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError('Certificate changed; check official Sber certificate instructions: '+name)
        path.write_bytes(data);certs.append(data)
    bundle=directory/'bundle.pem'
    bundle.write_bytes(Path(requests.certs.where()).read_bytes()+b'\n'+b'\n'.join(certs))
    return bundle


if __name__=='__main__':print(prepare())
