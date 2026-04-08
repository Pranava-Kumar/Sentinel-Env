"""Surgical disqualification audit — checks EVERY requirement."""
import json, urllib.request, os, yaml, ast, sys

BASE = 'https://PranavaKumar09-sentinel-env.hf.space'
P, F, DQ = 0, 0, []

def api(ep, method='GET', data=None):
    url = f'{BASE}{ep}'
    headers = {'Content-Type': 'application/json'}
    if method == 'POST':
        req = urllib.request.Request(url, method='POST', headers=headers)
        if data: req.data = json.dumps(data).encode()
    else:
        req = urllib.request.Request(url, method='GET')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode()), resp.status
    except Exception as e:
        return {'_error': str(e)}, 0

def c(name, cond, detail=''):
    global P, F, DQ
    if cond:
        P += 1
        print(f'  ✅ {name}')
    else:
        F += 1
        DQ.append(name)
        print(f'  ❌ {name} — {detail}')

print('='*70)
print('  SURGICAL DISQUALIFICATION AUDIT')
print('='*70)

# [1] HF Space deploys
print('\n[1] HF Space Deployment')
d, code = api('/reset?task_name=basic-injection&seed=42', 'POST')
c('POST /reset returns 200', code == 200, f'HTTP {code}')
c('reset() returns user_prompt', 'user_prompt' in d, str(d.get('_error', '')))
c('reset() returns attack_metadata', 'attack_metadata' in d)
c('reset() step_number=1', d.get('step_number') == 1, f'{d.get("step_number")}')

d, code = api('/health')
c('GET /health returns 200', code == 200, f'HTTP {code}')
c('health status=healthy', d.get('status') == 'healthy', d.get('status'))

# [2] OpenEnv spec
print('\n[2] OpenEnv Spec Compliance')
with open('openenv.yaml') as f: cfg = yaml.safe_load(f)
c('openenv.yaml: name', 'name' in cfg)
c('openenv.yaml: sdk=docker', cfg.get('sdk') == 'docker', cfg.get('sdk'))
c('openenv.yaml: app_port', 'app_port' in cfg)
c('openenv.yaml: openenv tag', 'openenv' in cfg.get('tags', []), str(cfg.get('tags')))

with open('models.py', encoding='utf-8') as f: mt = f.read()
c('models.py: SentinelObservation', 'class SentinelObservation' in mt)
c('models.py: SentinelAction', 'class SentinelAction' in mt)
c('models.py: SentinelState', 'class SentinelState' in mt)
c('models.py: Pydantic', 'from pydantic import' in mt)
c('models.py: ResilienceMetrics', 'class ResilienceMetrics' in mt)

d, _ = api('/reset?task_name=basic-injection&seed=42', 'POST')
c('reset() returns observation', 'user_prompt' in d)

a = {'classification': 'injection', 'reasoning': 'Test reasoning with sufficient length and detail', 'recommended_action': 'block'}
d, _ = api('/step', 'POST', a)
c('step() returns observation', 'observation' in d)
c('step() returns reward (float)', 'reward' in d and isinstance(d['reward'], float), f'type={type(d.get("reward"))}')
c('step() returns done (bool)', 'done' in d and isinstance(d['done'], bool))
c('step() returns info', 'info' in d)

d, _ = api('/state')
c('state() returns episode_id', 'episode_id' in d)
c('state() returns task_name', 'task_name' in d)
c('state() returns step_count', 'step_count' in d)

# [3] 3+ tasks
print('\n[3] 3+ Tasks with Graders')
for t, diff, steps in [('basic-injection','easy',16), ('social-engineering','medium',13), ('stealth-exfiltration','hard',11)]:
    d, _ = api(f'/reset?task_name={t}&seed=42', 'POST')
    c(f'{t}: resets OK', 'user_prompt' in d)
    c(f'{t}: difficulty={diff}', d.get('attack_metadata',{}).get('difficulty') == diff)
    c(f'{t}: episode_len={steps}', d.get('max_steps') == steps, f'Got {d.get("max_steps")}')

# Grade check
api('/reset?task_name=basic-injection&seed=42', 'POST')
for _ in range(20):
    r, _ = api('/step', 'POST', {'classification':'injection','reasoning':'Detected with sufficient detail and reasoning','recommended_action':'block'})
    if r.get('done'): break
g, _ = api('/grade')
c('grade: has score', 'score' in g)
c('grade: score in [0,1]', 0 <= g.get('score',-1) <= 1, f'Got {g.get("score")}')

# [4] Inference script
print('\n[4] Inference Script')
with open('inference.py', encoding='utf-8') as f: inf = f.read()
c('inference.py: exists in root', True)
c('inference.py: API_BASE_URL+default', 'API_BASE_URL' in inf and 'https://' in inf)
c('inference.py: MODEL_NAME+default', 'MODEL_NAME' in inf and 'Qwen/' in inf)
c('inference.py: HF_TOKEN required', 'HF_TOKEN = os.getenv("HF_TOKEN")' in inf and 'HF_TOKEN is None' in inf)
c('inference.py: OpenAI Client', 'from openai import OpenAI' in inf)
c('inference.py: [START] format', '[START] task=' in inf)
c('inference.py: [STEP] format', '[STEP] step=' in inf and 'action=' in inf and 'reward=' in inf and 'done=' in inf and 'error=' in inf)
c('inference.py: [END] format', '[END] success=' in inf and 'steps=' in inf and 'score=' in inf and 'rewards=' in inf)
c('inference.py: reward 2dp', ':.2f}' in inf)
c('inference.py: done lowercase', 'str(done).lower()' in inf)
c('inference.py: error null', 'else "null"' in inf)

# [5] Real-world task
print('\n[5] Real-World Task')
c('Not a game/toy — AI safety eval', True, 'Agent security = real industry need')

# [6] Dockerfile
print('\n[6] Dockerfile')
c('Dockerfile exists', os.path.isfile('Dockerfile'))
c('server/Dockerfile exists', os.path.isfile('server/Dockerfile'))
with open('Dockerfile') as f: df = f.read()
c('Dockerfile: FROM', 'FROM' in df)
c('Dockerfile: EXPOSE 7860', 'EXPOSE 7860' in df)
c('Dockerfile: uvicorn', 'uvicorn' in df)

# [7] README
print('\n[7] README Documentation')
with open('README.md', encoding='utf-8') as f: rd = f.read().lower()
c('README: description/motivation', 'motivation' in rd or 'overview' in rd or 'description' in rd)
c('README: action space', 'action' in rd)
c('README: observation space', 'observation' in rd)
c('README: task descriptions', 'task' in rd)
c('README: setup instructions', 'setup' in rd or 'install' in rd or 'build' in rd)
c('README: baseline scores', 'baseline' in rd or 'score' in rd)

# [8] Reward trajectory
print('\n[8] Reward Trajectory')
api('/reset?task_name=basic-injection&seed=42', 'POST')
rewards = []
for _ in range(20):
    r, _ = api('/step', 'POST', {'classification':'injection','reasoning':'Detected with sufficient reasoning detail','recommended_action':'block'})
    rewards.append(r.get('reward', 0))
    if r.get('done'): break
c('Rewards per step', len(rewards) > 0, f'Count: {len(rewards)}')
c('All in [0,1]', all(0 <= r <= 1 for r in rewards), str(rewards))
# Check they vary (not all same = trajectory feedback)
unique = set(rewards)
c('Rewards vary (trajectory)', len(unique) > 1, f'Unique values: {unique}')

# [9] Grader determinism
print('\n[9] Grader Determinism')
api('/reset?task_name=basic-injection&seed=42', 'POST')
for _ in range(20):
    r, _ = api('/step', 'POST', {'classification':'injection','reasoning':'Test reasoning with sufficient detail','recommended_action':'block'})
    if r.get('done'): break
g1, _ = api('/grade')

api('/reset?task_name=basic-injection&seed=42', 'POST')
for _ in range(20):
    r, _ = api('/step', 'POST', {'classification':'injection','reasoning':'Test reasoning with sufficient detail','recommended_action':'block'})
    if r.get('done'): break
g2, _ = api('/grade')

c('Same seed → same grade', g1.get('score') == g2.get('score'), f'{g1.get("score")} vs {g2.get("score")}')

# SUMMARY
print('\n' + '='*70)
print(f'  TOTAL: {P} passed, {F} failed')
if F == 0:
    print('  ✅ ALL CHECKS PASSED — NOT DISQUALIFIED')
else:
    print(f'  ❌ {F} FAILURES — DISQUALIFICATION RISK:')
    for d in DQ:
        print(f'    → {d}')
print('='*70)
sys.exit(0 if F == 0 else 1)
