"""Minimal OpenAI-compatible API bridge for two ARIS advisory roles; no executor."""
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
HOST='llm-a1n9j985xnxjqcqi.cn-beijing.maas.aliyuncs.com'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('API redirect refused; no credentials forwarded')


def read_secrets(path):
    path=Path(path)
    if not path.exists(): return {}
    if os.name!='nt' and path.stat().st_mode & 0o077:
        raise ValueError('Credentials file must have mode 0600')
    return json.loads(path.read_text(encoding='utf-8'))


def build_request(config, role, prompt, secrets, max_tokens=None):
    base=config['base_url'].rstrip('/')
    u=urllib.parse.urlparse(base)
    if u.scheme!='https' or u.hostname!=HOST or u.username or u.password or u.query or u.fragment or u.port not in (None,443):
        raise ValueError('Unapproved API endpoint')
    entry=config['roles'][role]
    key=os.environ.get(entry['key_env']) or secrets.get(entry['key_env'])
    if not key: raise ValueError('Missing credential for '+role)
    body={'model':entry['model'],'messages':[
        {'role':'system','content':entry.get('system','You are a research advisor.')},
        {'role':'user','content':prompt}],
        'max_tokens':max_tokens or config.get('max_tokens',4096),'stream':False}
    if 'reasoning_effort' in entry:
        body['reasoning_effort']=entry['reasoning_effort']
    return urllib.request.Request(base+'/chat/completions',data=json.dumps(body).encode('utf-8'),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=['planner','reviewer'],required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--prompt-file',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    cfg=json.loads((ROOT/'.aris/api_roles.json').read_text(encoding='utf-8'))
    prompt='Reply with the single word OK. This is a connection test only.' if a.check else a.prompt_file.read_text(encoding='utf-8')
    try:
        req=build_request(cfg,a.role,prompt,read_secrets(ROOT/'.aris/credentials.json'),max_tokens=512 if a.check else None)
        opener=urllib.request.build_opener(NoRedirect())
        with opener.open(req,timeout=cfg.get('timeout_seconds',60)) as resp:
            raw=json.loads(resp.read().decode('utf-8'))
        message=raw.get('choices',[{}])[0].get('message',{})
        content=message.get('content') or ''
        result={'role':a.role,'model':cfg['roles'][a.role]['model'],
            'status':'ok' if content.strip() else 'empty_content',
            'content':content,'usage':raw.get('usage'),
            'finish_reason':raw.get('choices',[{}])[0].get('finish_reason')}
        if not a.check: result['raw_response']=raw
        if a.output:
            # Never silently overwrite a previous review or connection record.
            a.output.parent.mkdir(parents=True,exist_ok=True)
            with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
        print(json.dumps(result,ensure_ascii=False))
        return 0 if result['status']=='ok' else 2
    except urllib.error.HTTPError as e:
        # Do not echo upstream bodies, headers, request objects, or credentials.
        print(json.dumps({'role':a.role,'status':'http_error','http_status':e.code})); return 2
    except (ValueError,KeyError,OSError,urllib.error.URLError) as e:
        print(json.dumps({'role':a.role,'status':'configuration_or_transport_error','error_type':type(e).__name__})); return 2


if __name__=='__main__': sys.exit(main())
