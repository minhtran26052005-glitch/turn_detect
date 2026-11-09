"""Bounded control through Jupyter's supported terminal API, no kernel reset.

An ephemeral terminal is closed after the command. It does not keep a runtime
alive, run training locally, or change any notebook cell. Credentials remain in
the existing ColabRuntime session and are never included in output.
"""
import argparse
import base64
import json
from pathlib import Path
import time
import uuid
import websocket
from colab_runtime import ColabRuntime


def execute(code,timeout=50):
    runtime=ColabRuntime(0);connection=runtime.connect()
    terminal=runtime.request('POST','api/terminals',json={}).json()['name']
    socket=None;nonce=uuid.uuid4().hex;begin='BEGIN_'+nonce;end='END_'+nonce
    error='ERROR_'+nonce
    wrapped=("print("+repr(begin)+",flush=True)\ntry:\n    exec("+repr(code)+")\n"
             "except BaseException:\n    import traceback\n    print("+repr(error)+",flush=True)\n    traceback.print_exc()\n"
             "finally:\n    print("+repr(end)+",flush=True)\n")
    encoded=base64.b64encode(wrapped.encode()).decode()
    command="python3 -c \"import base64;exec(base64.b64decode('"+encoded+"'))\"\n"
    try:
        url=runtime.base_url.replace('https://','wss://',1)+'terminals/websocket/'+terminal
        headers=[k+': '+v for k,v in runtime.http.headers.items() if k.lower() not in ('connection','accept-encoding','user-agent')]
        socket=websocket.create_connection(url,header=headers,timeout=10,origin=runtime.base_url.rstrip('/'))
        socket.send(json.dumps(['stdin',command]));output='';started=time.monotonic()
        while time.monotonic()-started<timeout:
            socket.settimeout(2)
            try:message=json.loads(socket.recv())
            except websocket.WebSocketTimeoutException:continue
            if message[0]=='stdout':output+=message[1]
            if begin in output and end in output:
                return {'connection':connection,'terminal_control':True,'remote_exception':error in output,'output':output.split(begin,1)[1].split(end,1)[0].strip()}
        raise TimeoutError('Terminal control command exceeded bound; no notebook kernel interrupt/reset')
    finally:
        if socket:socket.close()
        runtime.request('DELETE','api/terminals/'+terminal)
        runtime.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--code-file',required=True);a=p.parse_args()
    print(json.dumps(execute(Path(a.code_file).read_text(encoding='utf-8')),ensure_ascii=False))
