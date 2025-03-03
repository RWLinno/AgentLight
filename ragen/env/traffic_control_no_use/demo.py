from flask import Flask, jsonify, request, render_template_string
from .env_no_use import TrafficControlEnv
from ragen.utils import set_seed

app = Flask(__name__)
env = TrafficControlEnv()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traffic Control Demo</title>
    <style>
        body {
            font-family: monospace;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0;
            padding: 0;
        }
        #traffic-container {
            margin-top: 20px;
            width: 80%;
        }
        pre {
            background: #f4f4f4;
            padding: 10px;
            border: 1px solid #ccc;
            white-space: pre-wrap;
            width: 100%;
        }
        .controls {
            margin: 20px 0;
            display: flex;
            gap: 10px;
        }
        button {
            padding: 10px 15px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1>Traffic Control Demo</h1>
    
    <div class="controls">
        <button id="reset-btn">Reset</button>
        <div>
            <label for="action-select">Select Phase:</label>
            <select id="action-select">
                <option value="0">Phase 0</option>
                <option value="1">Phase 1</option>
                <option value="2">Phase 2</option>
                <option value="3">Phase 3</option>
                <option value="4">Phase 4</option>
                <option value="5">Phase 5</option>
                <option value="6">Phase 6</option>
                <option value="7">Phase 7</option>
            </select>
            <button id="step-btn">Apply</button>
        </div>
    </div>
    
    <div id="traffic-container">
        <h2>Traffic State</h2>
        <pre id="traffic-state">Loading...</pre>
        
        <h2>Statistics</h2>
        <div id="stats">
            <p>Reward: <span id="reward">0</span></p>
            <p>Time Step: <span id="time-step">0</span></p>
        </div>
    </div>

    <script>
        let timeStep = 0;
        
        async function resetEnv() {
            const response = await fetch('/reset');
            const data = await response.json();
            updateState(data.state);
            timeStep = 0;
            document.getElementById('time-step').textContent = timeStep;
            document.getElementById('reward').textContent = '0';
        }

        async function stepEnv() {
            const action = document.getElementById('action-select').value;
            const response = await fetch('/step', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ action: parseInt(action) }),
            });
            const data = await response.json();
            updateState(data.state);
            document.getElementById('reward').textContent = data.reward.toFixed(2);
            timeStep++;
            document.getElementById('time-step').textContent = timeStep;
            
            if (data.done) {
                alert('Simulation complete!');
                await resetEnv();
            }
        }

        function updateState(state) {
            document.getElementById('traffic-state').textContent = state;
        }

        document.getElementById('reset-btn').addEventListener('click', resetEnv);
        document.getElementById('step-btn').addEventListener('click', stepEnv);

        // Initialize
        resetEnv();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/reset', methods=['GET'])
def reset():
    state = env.reset()
    return jsonify({'state': state})

@app.route('/step', methods=['POST'])
def step():
    action = int(request.json.get('action', 0))
    state, reward, done, _ = env.step(action)
    return jsonify({'state': state, 'reward': reward, 'done': done})

@app.route('/close', methods=['GET'])
def close():
    return jsonify({'message': 'Environment closed'})

if __name__ == '__main__':
    app.run(debug=True)