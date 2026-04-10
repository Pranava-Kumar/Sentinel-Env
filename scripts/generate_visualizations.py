"""
Generate Graphviz/DOT and PlantUML visualizations from code review graph.
Outputs to .code-review-graph/visualizations/ folder.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent.parent / ".code-review-graph" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  GENERATING VISUALIZATIONS")
print("=" * 70)
print(f"  Output: {OUTPUT_DIR}")
print("=" * 70)


def run_graph_query(message: str) -> dict:
    """Run a graph query via MCP."""
    # This will be called manually since we can't use MCP directly in scripts
    pass


# ═══════════════════════════════════════════════════════════
# 1. Main Architecture Overview (DOT)
# ═══════════════════════════════════════════════════════════

print("\n[1/6] Generating main architecture DOT file...")

main_architecture_dot = '''digraph SentinelArchitecture {
    rankdir=TB;
    node [shape=box, style=filled, fontname="Arial"];
    edge [fontname="Arial", fontsize=10];
    
    // Graph styling
    graph [
        label="Sentinel Environment - Architecture Overview\\nMeta PyTorch Hackathon 2026",
        labelloc=t,
        fontsize=20,
        fontname="Arial Bold",
        bgcolor=white,
        pad=0.5,
        nodesep=0.8,
        ranksep=1.2
    ];
    
    // Legend
    subgraph cluster_legend {
        label="Legend";
        style=dashed;
        color=gray;
        
        legend_client [label="Client", fillcolor=lightblue];
        legend_server [label="Server", fillcolor=lightgreen];
        legend_env [label="Environment", fillcolor=lightyellow];
        legend_attacks [label="Attacks", fillcolor=salmon];
        legend_models [label="Models", fillcolor=plum];
        legend_inference [label="Inference", fillcolor=orange];
    }
    
    // Core modules as subgraphs
    subgraph cluster_inference {
        label="inference.py";
        style=filled;
        color=orange;
        fillcolor=lightyellow;
        
        inference_main [label="main()\\nEntry Point", shape=doubleoctagon, fillcolor=orange, fontcolor=white];
        run_single_task [label="run_single_task()\\nRuns 1 task", fillcolor=orange];
        get_model_response [label="get_model_response()\\nLLM API Call", fillcolor=orange];
        parse_model_response [label="parse_model_response()\\nJSON Parser", fillcolor=orange];
        
        inference_main -> run_single_task [label="calls 3x"];
        run_single_task -> get_model_response;
        get_model_response -> parse_model_response;
    }
    
    subgraph cluster_client {
        label="client.py";
        style=filled;
        color=lightblue;
        fillcolor=lightblue;
        
        sentinel_env [label="SentinelEnv\\nMain Client Class", shape=component, fillcolor=lightblue];
        client_reset [label="reset()\\nStart Episode"];
        client_step [label="step()\\nExecute Action"];
        client_close [label="close()\\nCleanup"];
        
        sentinel_env -> client_reset;
        sentinel_env -> client_step;
        sentinel_env -> client_close;
    }
    
    subgraph cluster_server {
        label="server/";
        style=filled;
        color=lightgreen;
        fillcolor=lightgreen;
        
        app [label="app.py\\nFastAPI App", shape=component, fillcolor=lightgreen];
        endpoint_reset [label="POST /reset"];
        endpoint_step [label="POST /step"];
        endpoint_grade [label="GET /grade"];
        endpoint_state [label="GET /state"];
        endpoint_health [label="GET /health"];
        
        app -> endpoint_reset;
        app -> endpoint_step;
        app -> endpoint_grade;
        app -> endpoint_state;
        app -> endpoint_health;
    }
    
    subgraph cluster_environment {
        label="sentinel_environment.py";
        style=filled;
        color=lightyellow;
        fillcolor=lightyellow;
        
        env_reset [label="reset()\\nInit Episode"];
        env_step [label="step()\\nProcess Action"];
        env_grade [label="get_episode_grade()"];
        env_state [label="state()"];
        
        env_reset -> env_step;
        env_step -> env_grade;
        env_reset -> env_state;
    }
    
    subgraph cluster_attacks {
        label="server/attacks/";
        style=filled;
        color=salmon;
        fillcolor=salmon;
        
        attack_engine [label="attack_engine.py\\nAttack Orchestrator", shape=component];
        mutation_engine [label="mutation_engine.py\\nPrompt Mutator"];
        jailbreak_loader [label="jailbreak_loader.py\\n114 Real Prompts"];
        basic_injections [label="basic_injections.py"];
        social_engineering [label="social_engineering.py"];
        stealth_exfiltration [label="stealth_exfiltration.py"];
        
        attack_engine -> mutation_engine;
        attack_engine -> jailbreak_loader;
        attack_engine -> basic_injections;
        attack_engine -> social_engineering;
        attack_engine -> stealth_exfiltration;
    }
    
    subgraph cluster_models {
        label="models.py";
        style=filled;
        color=plum;
        fillcolor=plum;
        
        pydantic_models [label="Pydantic Models\\nSentinelAction\\nSentinelObservation\\nThreatCategory", shape=note];
    }
    
    subgraph cluster_grader {
        label="grader.py";
        style=filled;
        color=cyan;
        fillcolor=cyan;
        
        grade_step [label="grade_step()\\nScore 1 Action"];
        grade_episode [label="grade_episode()\\nScore Episode"];
        
        grade_step -> grade_episode;
    }
    
    // Cross-module connections
    inference_main -> sentinel_env [label="initializes", color=blue];
    run_single_task -> client_reset [label="calls"];
    run_single_task -> client_step [label="calls N times"];
    client_reset -> endpoint_reset [label="HTTP POST", color=blue, style=dashed];
    client_step -> endpoint_step [label="HTTP POST", color=blue, style=dashed];
    endpoint_reset -> env_reset [label="routes"];
    endpoint_step -> env_step [label="routes"];
    endpoint_grade -> env_grade [label="routes"];
    env_step -> attack_engine [label="generates"];
    env_step -> grade_step [label="scores"];
    pydantic_models -> inference_main [label="imports", style=dotted];
    pydantic_models -> sentinel_env [label="imports", style=dotted];
    pydantic_models -> app [label="imports", style=dotted];
}'''

main_dot_file = OUTPUT_DIR / "01_architecture_overview.dot"
main_dot_file.write_text(main_architecture_dot, encoding="utf-8")
print(f"  ✓ Created: {main_dot_file.name}")


# ═══════════════════════════════════════════════════════════
# 2. Inference Pipeline Flow (DOT)
# ═══════════════════════════════════════════════════════════

print("\n[2/6] Generating inference pipeline DOT file...")

inference_flow_dot = '''digraph InferencePipeline {
    rankdir=LR;
    node [shape=box, style=filled, fontname="Arial", fontsize=12];
    edge [fontname="Arial", fontsize=10];
    
    graph [
        label="Inference Pipeline - 3 Task Execution Flow\\n[START] → [STEP] × N → [END] per task",
        labelloc=t,
        fontsize=18,
        fontname="Arial Bold",
        bgcolor=white
    ];
    
    // Start
    start [label="inference.py\\n__main__", shape=oval, fillcolor=green, fontcolor=white, fontsize=14];
    
    // Init phase
    subgraph cluster_init {
        label="Phase 1: Initialization";
        style=filled;
        color=lightyellow;
        
        init_llm [label="1. Initialize LLM Client\\nOpenAI(base_url, api_key)", fillcolor=lightblue];
        init_env [label="2. Initialize Environment\\nSentinelEnv(base_url)", fillcolor=lightgreen];
        
        init_llm -> init_env;
    }
    
    // Task loop
    subgraph cluster_task_loop {
        label="Phase 2: Task Loop (3 tasks)";
        style=filled;
        color=lightyellow;
        
        task_basic [label="Task 1: basic-injection", shape=octagon, fillcolor=orange, fontcolor=white];
        task_social [label="Task 2: social-engineering", shape=octagon, fillcolor=orange, fontcolor=white];
        task_stealth [label="Task 3: stealth-exfiltration", shape=octagon, fillcolor=orange, fontcolor=white];
        
        task_basic -> task_social -> task_stealth [style=dashed];
    }
    
    // Single task flow
    subgraph cluster_single_task {
        label="Single Task Execution";
        style=filled;
        color=lightyellow;
        
        reset [label="env.reset(task, seed)\\nReturns: episode_id, observation", fillcolor=lightgreen];
        step_loop [label="Step Loop (1 to MAX_STEPS)", shape=diamond, fillcolor=yellow];
        llm_call [label="get_model_response()\\nLLM API call → SentinelAction", fillcolor=lightblue];
        env_step [label="env.step(action)\\nReturns: obs, reward, done, info", fillcolor=lightgreen];
        log_step [label="_safe_log_step()\\n[STEP] output", fillcolor=gray];
        check_done [label="done?", shape=diamond, fillcolor=yellow];
        grade [label="env.client.get('/grade')\\nReturns: score, detection_rate", fillcolor=lightgreen];
        log_end [label="_safe_log_end()\\n[END] output", fillcolor=gray];
        
        reset -> step_loop;
        step_loop -> llm_call;
        llm_call -> env_step;
        env_step -> log_step;
        log_step -> check_done;
        check_done -> step_loop [label="no"];
        check_done -> grade [label="yes"];
        grade -> log_end;
    }
    
    // Summary
    summary [label="Print Summary\\n[SUMMARY] Tasks: 3/3\\n[PASS/FAIL] per task", shape=octagon, fillcolor=green, fontcolor=white];
    
    // End
    end [label="Cleanup & Exit\\navoid.close()\\ntracker.finish()", shape=oval, fillcolor=red, fontcolor=white];
    
    // Connections
    start -> init_llm;
    init_env -> task_basic;
    task_basic -> reset;
    task_social -> reset;
    task_stealth -> reset;
    log_end -> task_social [style=dotted, label="next task"];
    log_end -> task_stealth [style=dotted];
    log_end -> summary [label="all done"];
    summary -> end;
}'''

inference_dot_file = OUTPUT_DIR / "02_inference_pipeline.dot"
inference_dot_file.write_text(inference_flow_dot, encoding="utf-8")
print(f"  ✓ Created: {inference_dot_file.name}")


# ═══════════════════════════════════════════════════════════
# 3. Server Architecture (DOT)
# ═══════════════════════════════════════════════════════════

print("\n[3/6] Generating server architecture DOT file...")

server_arch_dot = '''digraph ServerArchitecture {
    rankdir=TB;
    node [shape=box, style=filled, fontname="Arial"];
    edge [fontname="Arial", fontsize=9];
    
    graph [
        label="Server Architecture - FastAPI Application\\nEndpoints, Middleware, Business Logic",
        labelloc=t,
        fontsize=18,
        fontname="Arial Bold",
        bgcolor=white
    ];
    
    // Endpoints
    subgraph cluster_endpoints {
        label="API Endpoints (app.py)";
        style=filled;
        color=lightblue;
        fillcolor=lightblue;
        
        ep_reset [label="POST /reset\\ntask_name, seed"];
        ep_step [label="POST /step\\nSentinelAction, X-Episode-ID"];
        ep_state [label="GET /state\\nX-Episode-ID"];
        ep_grade [label="GET /grade\\nX-Episode-ID"];
        ep_health [label="GET /health"];
        ep_profile [label="GET /resilience-profile"];
        ep_root [label="GET /\\nAPI Info"];
    }
    
    // Middleware
    subgraph cluster_middleware {
        label="Middleware (middleware.py)";
        style=filled;
        color=lightyellow;
        
        mw_structlog [label="structlog\\nJSON Logging"];
        mw_rate_limit [label="Rate Limiter\\n100 req/min"];
        mw_cors [label="CORS"];
    }
    
    // Core
    subgraph cluster_core {
        label="Core Business Logic";
        style=filled;
        color=lightgreen;
        
        episode_mgr [label="EpisodeManager\\n1000 concurrent episodes", shape=component];
        sentinel_env [label="SentinelEnvironment\\nEpisode State Machine", shape=component];
        attack_engine [label="AttackEngine\\nGenerates Attacks", shape=component];
        grader [label="Grader\\nscore_step(), grade_episode()", shape=component];
        reward_shaper [label="RewardShaper\\nReward Adjustment"];
    }
    
    // Attacks
    subgraph cluster_attacks {
        label="Attack Modules (server/attacks/)";
        style=filled;
        color=salmon;
        fillcolor=salmon;
        
        mutation [label="mutation_engine.py\\n120+ mutation strategies"];
        jailbreak [label="jailbreak_loader.py\\n114 real prompts"];
        basic [label="basic_injections.py"];
        social [label="social_engineering.py"];
        stealth [label="stealth_exfiltration.py"];
    }
    
    // Tracking
    subgraph cluster_tracking {
        label="Tracking & Metrics";
        style=filled;
        color=plum;
        fillcolor=plum;
        
        wandb [label="wandb_tracker.py\\nW&B Integration"];
        trackio [label="trackio_tracker.py\\nHF Spaces Tracking"];
        prometheus [label="Prometheus Metrics"];
    }
    
    // Flow
    ep_reset -> episode_mgr;
    ep_step -> episode_mgr;
    ep_grade -> episode_mgr;
    ep_state -> episode_mgr;
    
    episode_mgr -> sentinel_env;
    sentinel_env -> attack_engine;
    sentinel_env -> grader;
    grader -> reward_shaper;
    
    attack_engine -> mutation;
    attack_engine -> jailbreak;
    attack_engine -> basic;
    attack_engine -> social;
    attack_engine -> stealth;
    
    sentinel_env -> wandb;
    sentinel_env -> trackio;
    sentinel_env -> prometheus;
    
    mw_structlog -> ep_reset;
    mw_rate_limit -> ep_step;
}'''

server_dot_file = OUTPUT_DIR / "03_server_architecture.dot"
server_dot_file.write_text(server_arch_dot, encoding="utf-8")
print(f"  ✓ Created: {server_dot_file.name}")


# ═══════════════════════════════════════════════════════════
# 4. Client-Server Interaction Sequence (PlantUML)
# ═══════════════════════════════════════════════════════════

print("\n[4/6] Generating client-server sequence (PlantUML)...")

client_server_puml = '''@startuml ClientServerSequence
title Client-Server Interaction Sequence\\nOne Episode Execution

actor User
participant "inference.py" as Inference
participant "SentinelEnv\\nclient.py" as Client
participant "FastAPI Server\\napp.py" as Server
participant "EpisodeManager" as EpisodeMgr
participant "SentinelEnvironment" as Env
participant "AttackEngine" as Attacks
participant "Grader" as Grader

== Initialization Phase ==
User -> Inference: python inference.py
Inference -> Inference: Initialize LLM Client (OpenAI)
Inference -> Client: SentinelEnv(base_url)
activate Client
Client --> Inference: Client ready

== Task Loop (3 tasks) ==
loop For each task (basic-injection, social-engineering, stealth-exfiltration)
    Inference -> Inference: _safe_log_start([START])
    
    == Reset Episode ==
    Inference -> Client: reset(task_name, seed)
    Client -> Server: POST /reset?task_name=X&seed=Y
    activate Server
    Server -> EpisodeMgr: create_episode(task_name, seed)
    activate EpisodeMgr
    EpisodeMgr -> Env: new SentinelEnvironment()
    Env -> Attacks: generate_attack_sequence(task, seed)
    Attacks --> Env: Attack sequence
    EpisodeMgr --> Server: episode_id, observation
    Server --> Client: episode_id, observation
    Client --> Inference: SentinelObservation
    
    == Step Loop ==
    loop For each step (1 to MAX_STEPS)
        Inference -> Inference: get_model_response(obs.user_prompt)
        Inference -> Inference: LLM API call
        Inference -> Inference: SentinelAction
        
        Inference -> Client: step(action)
        Client -> Server: POST /step\\nbody: SentinelAction\\nheader: X-Episode-ID
        activate Server
        Server -> EpisodeMgr: get_episode(episode_id)
        EpisodeMgr --> Server: Environment instance
        Server -> Env: step(action)
        activate Env
        Env -> Attacks: get_current_attack()
        Attacks --> Env: Attack metadata
        Env -> Grader: grade_step(prediction, ground_truth)
        activate Grader
        Grader --> Env: reward, is_correct, is_partial
        deactivate Grader
        Env --> Server: observation, reward, done, info
        Server --> Client: step result
        Client --> Inference: obs, reward, done, info
        
        Inference -> Inference: _safe_log_step([STEP])
        
        alt done == true
            break
        end
    end
    
    == Grade Episode ==
    Inference -> Client: client.get("/grade", headers)
    Client -> Server: GET /grade\\nheader: X-Episode-ID
    Server -> Env: get_episode_grade()
    Env -> Grader: grade_episode(episode_results)
    Grader --> Env: score, detection_rate, etc.
    Env --> Server: grade result
    Server --> Client: grade JSON
    Client --> Inference: score, metrics
    
    Inference -> Inference: _safe_log_end([END])
end

== Summary ==
Inference -> Inference: Print [SUMMARY]
Inference -> Client: close()
Client -> Server: Connection close
deactivate Client
deactivate Server
Inference --> User: Exit (code 0)

@enduml'''

client_server_puml_file = OUTPUT_DIR / "04_client_server_sequence.puml"
client_server_puml_file.write_text(client_server_puml, encoding="utf-8")
print(f"  ✓ Created: {client_server_puml_file.name}")


# ═══════════════════════════════════════════════════════════
# 5. Execution Flows (PlantUML)
# ═══════════════════════════════════════════════════════════

print("\n[5/6] Generating execution flows (PlantUML)...")

execution_flows_puml = '''@startuml ExecutionFlows
title Top Execution Flows - Sentinel Environment

skinparam backgroundColor white
skinparam shadowing false

package "inference.py" as Inference {
    [main] as main_inf
    [run_single_task] as run_task
    [get_model_response] as llm_call
}

package "client.py" as Client {
    [SentinelEnv.__aenter__] as client_init
    [SentinelEnv.reset] as client_reset
    [SentinelEnv.step] as client_step
}

package "server/app.py" as Server {
    [POST /reset] as ep_reset
    [POST /step] as ep_step
    [GET /grade] as ep_grade
}

package "server/episode_manager.py" as EpMgr {
    [EpisodeManager.create_episode] as create_ep
    [EpisodeManager.get_episode] as get_ep
}

package "server/sentinel_environment.py" as Env {
    [SentinelEnvironment.reset] as env_reset
    [SentinelEnvironment.step] as env_step
    [SentinelEnvironment.get_episode_grade] as env_grade
}

package "server/grader.py" as Grader {
    [grade_step] as grade_step_fn
    [grade_episode] as grade_ep_fn
}

package "server/attack_engine.py" as Attacks {
    [generate_attack_sequence] as gen_attacks
    [get_current_attack] as get_attack
}

' Flow 1: Reset
main_inf --> run_task : calls (3x)
run_task --> client_reset : await
client_reset --> ep_reset : HTTP POST
ep_reset --> create_ep : call
create_ep --> env_reset : new instance
env_reset --> gen_attacks : load attacks
gen_attacks --> env_reset : attack_sequence

' Flow 2: Step
run_task --> llm_call : get action
llm_call --> run_task : SentinelAction
run_task --> client_step : await
client_step --> ep_step : HTTP POST + X-Episode-ID
ep_step --> get_ep : lookup episode
get_ep --> env_step : call
env_step --> get_attack : current attack
get_attack --> env_step : attack metadata
env_step --> grade_step_fn : score prediction
grade_step_fn --> env_step : reward, is_correct
env_step --> ep_step : obs, reward, done, info

' Flow 3: Grade
run_task --> ep_grade : HTTP GET
ep_grade --> get_ep : lookup
get_ep --> env_grade : call
env_grade --> grade_ep_fn : aggregate results
grade_ep_fn --> env_grade : score, metrics
env_grade --> ep_grade : JSON response

note right of main_inf
    Execution starts here
    Loops through 3 tasks:
    - basic-injection
    - social-engineering
    - stealth-exfiltration
end note

note right of grade_step_fn
    Scoring:
    - Correct: +0.6
    - Partial: +0.3
    - Binary: +0.15
    - Missed: -0.5
    - False positive: -0.3
    - Reasoning: up to +0.2
end note

@enduml'''

exec_flows_puml_file = OUTPUT_DIR / "05_execution_flows.puml"
exec_flows_puml_file.write_text(execution_flows_puml, encoding="utf-8")
print(f"  ✓ Created: {exec_flows_puml_file.name}")


# ═══════════════════════════════════════════════════════════
# 6. Module Dependencies (DOT)
# ═══════════════════════════════════════════════════════════

print("\n[6/6] Generating module dependency graph (DOT)...")

deps_dot = '''digraph ModuleDependencies {
    rankdir=LR;
    node [shape=box, style=filled, fontname="Arial", fontsize=11];
    edge [fontname="Arial", fontsize=9];
    
    graph [
        label="Module Dependency Graph\\nSolid = import, Dashed = runtime call",
        labelloc=t,
        fontsize=16,
        fontname="Arial Bold",
        bgcolor=white,
        compound=true
    ];
    
    // Root modules
    inference [label="inference.py", fillcolor=orange, fontcolor=white];
    client [label="client.py", fillcolor=lightblue];
    models [label="models.py", fillcolor=plum];
    inference_logging [label="inference_logging.py", fillcolor=gray];
    
    // Server modules
    server_app [label="server/app.py", fillcolor=lightgreen];
    server_env [label="server/sentinel_environment.py", fillcolor=lightyellow];
    server_ep_mgr [label="server/episode_manager.py", fillcolor=lightyellow];
    server_grader [label="server/grader.py", fillcolor=cyan];
    server_attacks [label="server/attack_engine.py", fillcolor=salmon];
    server_reward [label="server/reward_shaper.py", fillcolor=lightyellow];
    server_resilience [label="server/resilience_profile.py", fillcolor=lightyellow];
    
    // Attack modules
    mutation [label="server/attacks/mutation_engine.py", fillcolor=salmon];
    jailbreak [label="server/attacks/jailbreak_loader.py", fillcolor=salmon];
    basic_inj [label="server/attacks/basic_injections.py", fillcolor=salmon];
    social [label="server/attacks/social_engineering.py", fillcolor=salmon];
    stealth [label="server/attacks/stealth_exfiltration.py", fillcolor=salmon];
    
    // Tracking
    wandb [label="server/wandb_tracker.py", fillcolor=plum];
    trackio [label="server/trackio_tracker.py", fillcolor=plum];
    
    // Dependencies (imports)
    inference -> client [label="imports"];
    inference -> models [label="imports"];
    inference -> inference_logging [label="imports"];
    inference -> wandb [label="imports"];
    
    client -> models [label="imports"];
    
    server_app -> models [label="imports"];
    server_app -> server_ep_mgr [label="imports"];
    server_app -> server_attacks [label="imports"];
    
    server_env -> server_grader [label="imports"];
    server_env -> server_attacks [label="imports"];
    server_env -> server_resilience [label="imports"];
    server_env -> wandb [label="imports"];
    server_env -> trackio [label="imports"];
    
    server_ep_mgr -> server_env [label="imports"];
    
    server_attacks -> mutation [label="imports"];
    server_attacks -> jailbreak [label="imports"];
    server_attacks -> basic_inj [label="imports"];
    server_attacks -> social [label="imports"];
    server_attacks -> stealth [label="imports"];
    
    server_reward -> server_grader [label="imports"];
    
    // Runtime calls (dashed)
    inference -> server_app [label="HTTP calls", style=dashed, color=blue];
    client -> server_app [label="HTTP calls", style=dashed, color=blue];
}'''

deps_dot_file = OUTPUT_DIR / "06_module_dependencies.dot"
deps_dot_file.write_text(deps_dot, encoding="utf-8")
print(f"  ✓ Created: {deps_dot_file.name}")


# ═══════════════════════════════════════════════════════════
# 7. README with visualization instructions
# ═══════════════════════════════════════════════════════════

print("\n[7/7] Creating README with visualization instructions...")

readme_content = """# Visualizations - Sentinel Environment

This folder contains Graphviz/DOT and PlantUML files for visualizing the code review graph.

## 📊 Generated Files

| File | Type | Description |
|------|------|-------------|
| `01_architecture_overview.dot` | Graphviz | Main architecture overview |
| `02_inference_pipeline.dot` | Graphviz | 3-task inference execution flow |
| `03_server_architecture.dot` | Graphviz | Server endpoint and module structure |
| `04_client_server_sequence.puml` | PlantUML | Client-server interaction sequence |
| `05_execution_flows.puml` | PlantUML | Top execution flows through codebase |
| `06_module_dependencies.dot` | Graphviz | Module import and runtime dependencies |

---

## 🎨 How to Visualize

### **Method 1: Graphviz Online (Easiest)**

1. Go to: **https://dreampuf.github.io/GraphvizOnline/**
2. Open any `.dot` file in a text editor
3. Copy the entire content
4. Paste into the online editor
5. View and interact with the graph

### **Method 2: PlantUML Online**

1. Go to: **https://www.plantuml.com/plantuml/uml/**
2. Open any `.puml` file in a text editor
3. Copy the entire content
4. Paste into the online editor (click "Submit")
5. View the diagram

### **Method 3: Local Graphviz Installation**

```bash
# Install Graphviz
# Windows (using winget):
winget install graphviz

# Or using Chocolatey:
choco install graphviz

# Or download from: https://graphviz.org/download/

# Render DOT files to PNG:
dot -Tpng 01_architecture_overview.dot -o architecture.png
dot -Tpng 02_inference_pipeline.dot -o inference.png
dot -Tpng 03_server_architecture.dot -o server.png
dot -Tpng 06_module_dependencies.dot -o dependencies.png

# Open the generated images:
start architecture.png
```

### **Method 4: VS Code Extensions**

1. Install **"Graphviz Preview"** extension
2. Open any `.dot` file
3. Press `Ctrl+Shift+V` to preview

For PlantUML:
1. Install **"PlantUML"** extension
2. Open any `.puml` file
3. Press `Alt+D` to preview

### **Method 5: Python with graphviz package**

```bash
pip install graphviz
```

```python
from graphviz import Source

# Render and view
src = Source.from_file("01_architecture_overview.dot")
src.render("architecture", format="png", cleanup=True)
```

---

## 📋 Quick Start (Recommended)

For the fastest way to see the graphs:

1. **For DOT files**: Visit https://dreampuf.github.io/GraphvizOnline/
2. **For PlantUML files**: Visit https://www.plantuml.com/plantuml/uml/
3. Copy-paste and view instantly!

---

## 📖 What Each Diagram Shows

### 01 - Architecture Overview
High-level view of all major components and their relationships. Shows how inference.py, client.py, server endpoints, attack engine, and grader interact.

### 02 - Inference Pipeline
Detailed flow of the inference.py execution: initialization → 3-task loop → step loop → grading → summary.

### 03 - Server Architecture
FastAPI application structure showing endpoints, middleware, episode manager, attack modules, and tracking integrations.

### 04 - Client-Server Sequence
UML sequence diagram showing the complete interaction flow from inference startup through episode completion.

### 05 - Execution Flows
UML activity diagram showing the top execution paths through the codebase with scoring logic.

### 06 - Module Dependencies
Import and runtime dependency graph showing how modules depend on each other.

---

## 🔒 Note

This folder is gitignored to avoid cluttering the repository with generated visualization files. The source code review graph remains in `.code-review-graph/` directory.
"""

readme_file = OUTPUT_DIR / "README.md"
readme_file.write_text(readme_content, encoding="utf-8")
print(f"  ✓ Created: {readme_file.name}")


# ═══════════════════════════════════════════════════════════
# 8. Update .gitignore
# ═══════════════════════════════════════════════════════════

print("\n[8/8] Updating .gitignore...")

gitignore_file = Path(__file__).parent.parent / ".gitignore"
gitignore_content = gitignore_file.read_text(encoding="utf-8", errors="ignore") if gitignore_file.exists() else ""

visualizations_ignore = """
# Visualization exports (generated, not tracked)
.code-review-graph/visualizations/
"""

if ".code-review-graph/visualizations/" not in gitignore_content:
    gitignore_file.write_text(gitignore_content + visualizations_ignore, encoding="utf-8")
    print(f"  ✓ Added .code-review-graph/visualizations/ to .gitignore")
else:
    print(f"  ✓ .code-review-graph/visualizations/ already in .gitignore")


# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  VISUALIZATION GENERATION COMPLETE")
print("=" * 70)
print(f"\n  Files created: {len(list(OUTPUT_DIR.glob('*')))}")
print(f"  Output directory: {OUTPUT_DIR}")
print("\n  📊 How to view:")
print("    1. DOT files:  https://dreampuf.github.io/GraphvizOnline/")
print("    2. PlantUML:   https://www.plantuml.com/plantuml/uml/")
print("    3. Local:      dot -Tpng <file.dot> -o <file.png>")
print("\n" + "=" * 70)
