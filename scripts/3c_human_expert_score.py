"""
(3c/4)

Human Expert Score Evaluation via pairwise comparison of candidate solutions.

Prerequisites:
openai >= 1.0.0
xgboost >= 0.90
numpy >= 1.17.2
pandas >= 0.25.1
sklearn >= 0.22.1
json

INSTRUCTIONS:

firstly run:
python 3c_human_expert_score.py

ssh: 
ssh -L 5000:localhost:5000 ubuntu@z1.cloud.garaza.io -t ssh -L 5000:localhost:5000 bluefield-z1

and then the website will be available at (on local machine):
http://localhost:5000

"""

import re
import json
import random
import secrets
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for

app.secret_key = "trivial-secret-for-faculty-testing"  # enabling Flask session cookies

TESTER_IDS = ["tester1", "tester2", "tester3", "tester4", "tester5"]
COMPARISONS_PER_TESTER = 30
SIMPLE_PASSCODE = "network2026"  # shared passcode to enable Flask session cookies

app = Flask(__name__)

BASE_DIR = Path("json_log")

GROUNDTRUTH_DIR = BASE_DIR / "1_groundtruth_and_xgboost_prediction"

OPENAI_DIR = BASE_DIR / "2_openai_evaluation"
OLLAMA_DIR = BASE_DIR / "2_ollama_evaluation"
RETRAINED_DIR = BASE_DIR / "2_retrained_evaluation"

RESULT_DIR = (
    BASE_DIR
    / "3_evaluation_results"
    / "3_human_expert_score"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

SESSION_FILE = (
    RESULT_DIR
    / f"human_evaluation_session_{SESSION_ID}.json"
)

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Login</title></head>
<body style="font-family: Arial; max-width: 400px; margin: 80px auto;">
  <h2>Human Expert Evaluation Login</h2>
  <form method="POST" action="/login">
    <label>Tester ID:</label><br>
    <select name="tester_id">
      {% for t in tester_ids %}
        <option value="{{ t }}">{{ t }}</option>
      {% endfor %}
    </select><br><br>
    <label>Passcode:</label><br>
    <input type="password" name="passcode"><br><br>
    <button type="submit">Enter</button>
  </form>
  {% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
</body>
</html>
"""

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Human Expert Evaluation</title>

<style>

.panel h1,
.panel h2,
.panel h3,
.panel h4 {
    margin-top: 1em;
}

.panel p {
    line-height: 1.5;
}

.panel ul,
.panel ol {
    padding-left: 25px;
}

.panel code {
    background: #eeeeee;
    padding: 2px 4px;
    border-radius: 4px;
}

.panel pre {
    background: #f0f0f0;
    padding: 10px;
    border-radius: 6px;
    overflow-x: auto;
}

.panel blockquote {
    border-left: 4px solid #cccccc;
    padding-left: 12px;
    color: #666666;
}

body {
    font-family: Arial, sans-serif;
    margin: 30px;
    background: #f5f5f5;
}

.header {
    background: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}

.log-box {
    display: none;
    margin-top: 15px;
    background: #272822;
    color: #f8f8f2;
    padding: 15px;
    border-radius: 8px;
    overflow-x: auto;
    max-height: 400px;
    font-size: 13px;
}

.toggle-btn {
    padding: 8px 16px;
    margin-top: 10px;
}

.container {
    display: flex;
    gap: 20px;
}

.panel {
    flex: 1;
    background: white;
    border-radius: 10px;
    padding: 20px;
    overflow-y: auto;
    max-height: 75vh;
}

.buttons {
    text-align: center;
    margin-top: 20px;
}

button {
    font-size: 18px;
    padding: 12px 24px;
    margin: 10px;
    cursor: pointer;
}

pre {
    white-space: pre-wrap;
    word-wrap: break-word;
}

.progress {
    margin-bottom: 15px;
    font-size: 18px;
    font-weight: bold;
}

.panel.unavailable {
    opacity: 0.5;
}

</style>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>

<body>

<div class="progress" id="progress"></div>

<div class="header">
    <h2>Sample Information</h2>

    <p><strong>Ground Truth:</strong> <span id="groundtruth"></span></p>
    <p><strong>XGBoost Prediction:</strong> <span id="prediction"></span></p>

    <button class="toggle-btn" onclick="toggleLog()">
        Show Input JSON Log
    </button>

    <pre id="original_log" class="log-box"></pre>
</div>

<div class="container">

    <div class="panel">
        <h2>Candidate A</h2>

        <h3>Reasoning</h3>
        <div id="a_reasoning"></div>


        <h3>Solution</h3>
        <div id="a_solution"></div>
    </div>

    <div class="panel">
        <h2>Candidate B</h2>

        <h3>Reasoning</h3>
        <div id="b_reasoning"></div>

        <h3>Solution</h3>
        <div id="b_solution"></div>
    </div>

    <div class="panel" id="panel-c">
        <h2 id="panel-c-title">Candidate C</h2>

        <h3>Reasoning</h3>
        <div id="c_reasoning"></div>

        <h3>Solution</h3>
        <div id="c_solution"></div>
    </div>

</div>

<div class="buttons">
    <button onclick="vote('A')">Choose A</button>
    <button onclick="vote('B')">Choose B</button>
    <button id="button-c" onclick="vote('C')">Choose C</button>
</div>

<script>

let tasks = [];
let current = 0;

function toggleLog() {

    const box = document.getElementById("original_log");

    if(box.style.display === "block")
        box.style.display = "none";
    else
        box.style.display = "block";
}

async function loadTasks() {

    const response = await fetch('/tasks');
    tasks = await response.json();

    render();
}

function render() {

    if(current >= tasks.length) {

        document.body.innerHTML =
            "<h1>All evaluations completed.</h1>";

        return;
    }

    const t = tasks[current];

    const buttonC = document.getElementById("button-c");
    const panelC = document.getElementById("panel-c");
    const panelCTitle = document.getElementById("panel-c-title");

    if (t.c.missing) {

        buttonC.disabled = true;
        buttonC.innerText = "Unavailable";

        panelC.classList.add("unavailable");
        panelCTitle.innerText = "Candidate C (Not available)";

        document.getElementById("c_reasoning").innerHTML =
            "<i>No third evaluation was generated.</i>";

        document.getElementById("c_solution").innerHTML =
            "<i>No explanation available.</i>";

    } else {

        buttonC.disabled = false;
        buttonC.innerText = "Choose C";

        panelC.classList.remove("unavailable");
        panelCTitle.innerText = "Candidate C";
    }

    document.getElementById("progress").innerText =
        `Comparison ${current + 1} / ${tasks.length}`;

    document.getElementById("groundtruth").innerText =
        t.ground_truth;

    document.getElementById("prediction").innerText =
        t.prediction;

    document.getElementById("a_reasoning").innerHTML =
    marked.parse(t.a.reasoning || "");

    document.getElementById("a_solution").innerHTML =
        marked.parse(t.a.solution || "");

    document.getElementById("b_reasoning").innerHTML =
        marked.parse(t.b.reasoning || "");

    document.getElementById("b_solution").innerHTML =
        marked.parse(t.b.solution || "");

    document.getElementById("c_reasoning").innerHTML =
        marked.parse(t.c.reasoning || "");

    document.getElementById("c_solution").innerHTML =
        marked.parse(t.c.solution || "");

    document.getElementById("original_log").textContent =
        JSON.stringify(t.original_log, null, 2);

    document.getElementById("original_log").style.display = "none";
}

async function vote(choice) {

    const task = tasks[current];

    await fetch('/vote', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            task: task,
            winner: choice
        })
    });

    current++;
    render();
}

document.addEventListener('keydown', function(e) {

    if(e.key === '1')
        vote('A');

    if(e.key === '2')
        vote('B');

    if(e.key === '3' && !tasks[current].c.missing)
        vote('C');
});

loadTasks();

</script>

</body>
</html>
"""

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        tester_id = request.form.get("tester_id")
        passcode = request.form.get("passcode")

        if passcode != SIMPLE_PASSCODE or tester_id not in TESTER_IDS:
            return render_template_string(LOGIN_HTML, tester_ids=TESTER_IDS, error="Invalid tester ID or passcode.")

        session["tester_id"] = tester_id
        return redirect(url_for("index"))

    return render_template_string(LOGIN_HTML, tester_ids=TESTER_IDS, error=None)


@app.route("/")
def index():
    if "tester_id" not in session:
        return redirect(url_for("login"))
    return render_template_string(HTML)


def extract_sample_id(filename):
    match = re.search(r'(\d{8}_\d{6}_\d+)', filename)
    return match.group(1) if match else None

def load_original_log(sample_id):
    path = GROUNDTRUTH_DIR / f"prediction_{sample_id}.json"

    if not path.exists():
        return ["FILE NOT FOUND"]

    return load_json(path)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_latest_sample_ids():

    sample_ids = set()

    for folder in [OPENAI_DIR, OLLAMA_DIR, RETRAINED_DIR]:

        if not folder.exists():
            continue

        for f in folder.glob("evaluation_*.json"):

            sid = extract_sample_id(f.name)

            if sid:
                sample_ids.add(sid)

    return sorted(sample_ids)


def load_candidate(folder, sample_id):

    path = folder / f"evaluation_{sample_id}.json"

    if not path.exists():
        return None

    try:
        data = load_json(path)

        if not data:
            return None

        entry = data[0]

        return {
            "model": entry.get("model", "unknown"),
            "reasoning": entry.get("reasoning", ""),
            "solution": entry.get("solution", ""),
            "missing": False
        }

    except Exception:
        return None


def build_tasks(assigned_sample_ids):
    tasks = []
    for sample_id in assigned_sample_ids:

        gt_path = (
            GROUNDTRUTH_DIR
            / f"ground_truth_{sample_id}.json"
        )

        pred_path = (
            GROUNDTRUTH_DIR
            / f"prediction_{sample_id}.json"
        )

        if not gt_path.exists() or not pred_path.exists():
            continue

        try:
            gt = load_json(gt_path)
            pred = load_json(pred_path)
        except Exception:
            continue

        candidates = []

        openai = load_candidate(OPENAI_DIR, sample_id)
        ollama = load_candidate(OLLAMA_DIR, sample_id)
        retrained = load_candidate(RETRAINED_DIR, sample_id)

        for c in [openai, ollama, retrained]:
            if c:
                candidates.append(c)

        if len(candidates) < 2:
            print("Skipping sample_id", sample_id, "because fewer than two candidates exist")
            continue

        random.shuffle(candidates)

        while len(candidates) < 3:
            candidates.append({
                "model": "No candidate available",
                "reasoning": (
                    "This candidate was not generated for this evaluation."
                ),
                "solution": (
                    "No explanation is available."
                ),
                "missing": True
            })

        task = {
            "sample_id": sample_id,
            "ground_truth": gt.get(
                "most_common_true_label",
                "Unknown"
            ),
            "prediction": pred.get(
                "predicted_class_label",
                pred.get(
                    "model_prediction",
                    "Unknown"
                )
            ),
            "original_log": load_original_log(sample_id),
        }

        letters = ["a", "b", "c"]

        for i, candidate in enumerate(candidates):
            task[letters[i]] = candidate

        tasks.append(task)

    return tasks


@app.route("/")
def index():
    return render_template_string(HTML)

def get_tester_sample_ids(tester_id, all_sample_ids):
    if tester_id not in TESTER_IDS:
        return []
    idx = TESTER_IDS.index(tester_id)
    start = idx * COMPARISONS_PER_TESTER
    end = start + COMPARISONS_PER_TESTER
    return all_sample_ids[start:end]

@app.route("/tasks")
def tasks():
    tester_id = session.get("tester_id")
    if tester_id is None:
        return jsonify({"error": "not logged in"}), 401

    all_sample_ids = find_latest_sample_ids()
    assigned = get_tester_sample_ids(tester_id, all_sample_ids)
    return jsonify(build_tasks(assigned))


@app.route("/vote", methods=["POST"])
def vote():

    data = request.json

    task = data["task"]
    winner = data["winner"]

    if SESSION_FILE.exists():

        with open(SESSION_FILE, "r") as f:
            session_data = json.load(f)

    else:

        session_data = {
            "session_id": SESSION_ID,
            "created": datetime.now().isoformat(),
            "comparisons": []
        }

    session_data["comparisons"].append({
        "timestamp": datetime.now().isoformat(),

        "sample_id": task["sample_id"],

        "ground_truth": task["ground_truth"],
        "xgboost_prediction": task["prediction"],

        "candidate_a_model": task["a"]["model"],
        "candidate_b_model": task["b"]["model"],
        "candidate_c_model": task["c"]["model"],

        "winner": winner,
        "winner_model": {
            "A": task["a"]["model"],
            "B": task["b"]["model"],
            "C": task["c"]["model"],
        }[winner]
    })

    with open(SESSION_FILE, "w") as f:
        json.dump(session_data, f, indent=2)

    return jsonify({"status": "ok"})

def get_session_file(tester_id):
    return RESULT_DIR / f"human_evaluation_session_{tester_id}_{SESSION_ID}.json"


@app.route("/vote", methods=["POST"])
def vote():
    tester_id = session.get("tester_id")
    if tester_id is None:
        return jsonify({"error": "not logged in"}), 401

    data = request.json
    task = data["task"]
    winner = data["winner"]

    session_file = get_session_file(tester_id)

    if session_file.exists():
        with open(session_file, "r") as f:
            session_data = json.load(f)
    else:
        session_data = {
            "session_id": SESSION_ID,
            "tester_id": tester_id,
            "created": datetime.now().isoformat(),
            "comparisons": []
        }

    session_data["comparisons"].append({
        "timestamp": datetime.now().isoformat(),
        "tester_id": tester_id,          # NEW
        "sample_id": task["sample_id"],
        "ground_truth": task["ground_truth"],
        "xgboost_prediction": task["prediction"],
        "candidate_a_model": task["a"]["model"],
        "candidate_b_model": task["b"]["model"],
        "candidate_c_model": task["c"]["model"],
        "winner": winner,
        "winner_model": {
            "A": task["a"]["model"],
            "B": task["b"]["model"],
            "C": task["c"]["model"],
        }[winner]
    })

    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)

    return jsonify({"status": "ok"})

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )