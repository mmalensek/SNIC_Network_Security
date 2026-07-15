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
from pathlib import Path
from datetime import datetime
from itertools import combinations
from flask import Flask, render_template_string, request, jsonify

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
        Show Original JSON Log
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

</div>

<div class="buttons">
    <button onclick="vote('A')">A Better</button>
    <button onclick="vote('TIE')">Tie</button>
    <button onclick="vote('B')">B Better</button>
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

    if(e.key === 'ArrowLeft')
        vote('A');

    if(e.key === 'ArrowRight')
        vote('B');

    if(e.key === 'ArrowDown')
        vote('TIE');
});

loadTasks();

</script>

</body>
</html>
"""


def extract_sample_id(filename):
    match = re.search(r'(\d{8}_\d{6}_\d+)', filename)
    return match.group(1) if match else None

def latest_prediction_batch():

    files = sorted(GROUNDTRUTH_DIR.glob("prediction_*.json"))

    if not files:
        return None

    latest = max(
        files,
        key=lambda p: extract_sample_id(p.name)
    )

    sample_id = extract_sample_id(latest.name)

    # remove the trailing "_n"
    return sample_id.rsplit("_", 1)[0]


def load_original_log(sample_id):

    batch = latest_prediction_batch()

    if batch is None:
        return ["BATCH NOT FOUND"]

    number = sample_id.rsplit("_", 1)[1]

    path = (
        GROUNDTRUTH_DIR
        / f"prediction_{batch}_{number}.json"
    )

    if not path.exists():
        return ["FILE NOT FOUND"]

    data = load_json(path)

    print(data.keys())
    return data

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
            "solution": entry.get("solution", "")
        }

    except Exception:
        return None


def build_tasks():

    tasks = []

    for sample_id in find_latest_sample_ids():

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
            continue

        for left, right in combinations(candidates, 2):

            pair = [left, right]
            random.shuffle(pair)

            tasks.append({
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

                "a": pair[0],
                "b": pair[1]
            })

    return tasks


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/tasks")
def tasks():

    return jsonify(build_tasks())


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

        "winner": winner
    })

    with open(SESSION_FILE, "w") as f:
        json.dump(session_data, f, indent=2)

    return jsonify({"status": "ok"})


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )