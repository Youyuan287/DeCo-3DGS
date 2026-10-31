import os, random, sys
S = int(sys.argv[1]); script = os.path.abspath(sys.argv[2]); args = sys.argv[3:]
import numpy as np, torch
random.seed(S); np.random.seed(S); torch.manual_seed(S); torch.cuda.manual_seed_all(S)
sys.path.insert(0, os.path.dirname(script)); sys.path.insert(0, os.getcwd())
sys.argv = [script] + args
print("[run_seeded] seed=%d script=%s cwd=%s" % (S, script, os.getcwd()), flush=True)
g = {"__name__": "__main__", "__file__": script}
exec(compile(open(script).read(), script, "exec"), g)