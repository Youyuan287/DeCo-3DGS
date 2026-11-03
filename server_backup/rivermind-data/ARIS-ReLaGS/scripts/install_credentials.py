"""Read the two user-authorized API keys from stdin; never print their values."""
import json
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
data=json.load(sys.stdin)
assert set(data)=={'ARIS_GLM_API_KEY','ARIS_DEEPSEEK_API_KEY'}
assert all(isinstance(v,str) and v.strip() for v in data.values())
private=root/'.aris'
private.mkdir(exist_ok=True)
if os.name!='nt': private.chmod(0o700)
dest=private/'credentials.json'
# Exclusive creation: explicit rotation is required instead of silent overwrite.
fd=os.open(str(dest),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w',encoding='utf-8') as f:
    json.dump(data,f)
print('Credentials installed; values omitted')
