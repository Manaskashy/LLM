import os
import csv
import json
from flask import Flask, request, render_template_string
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from a .env file in the project root
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# --- Groq API Setup ---
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please create a .env file and add your key.")
client = Groq(api_key=api_key)

# --- Configuration ---
CSV_FILE = 'call_analysis.csv'
# UPDATED MODEL NAME to an instant, fast model (overridable via GROQ_MODEL)
MODEL_NAME = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

def save_to_csv(transcript, summary, sentiment):
    """Appends the analysis output to a CSV file."""
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Transcript', 'Summary', 'Sentiment'])
        writer.writerow([transcript, summary, sentiment])

# --- Core Logic ---
def analyze_transcript(transcript):
    """Uses the Groq API to summarize and analyze sentiment."""
    system_prompt = """
    You are an expert AI assistant for analyzing customer call transcripts.
    Your task is to provide a summary and sentiment analysis.
    The user will provide a transcript, and you must return ONLY a JSON object 
    with two keys: 'summary' and 'sentiment'.
    - The 'summary' should be 2-3 sentences long.
    - The 'sentiment' must be one of the following strings: 'Positive', 'Neutral', or 'Negative'.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript}
            ],
            model=MODEL_NAME,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        response_content = chat_completion.choices[0].message.content
        analysis_result = json.loads(response_content)
        return analysis_result.get('summary'), analysis_result.get('sentiment')
    except Exception as e:
        error_message = f"An error occurred with the Groq API: {e}"
        print(error_message)
        return error_message, "Unknown"

# --- Web Interface (UI) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transcript Analyzer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #4a90e2;
            --background-color: #f7f9fc;
            --card-background: #ffffff;
            --text-color: #333d49;
            --border-color: #e0e6ed;
            --shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        [data-theme="dark"] {
            --primary-color: #7ab8ff;
            --background-color: #0f172a;
            --card-background: #111827;
            --text-color: #e5e7eb;
            --border-color: #1f2937;
            --shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }
        body {
            font-family: 'Poppins', sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            margin: 0;
            padding: 2rem;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            min-height: 100vh;
        }
        .container {
            width: 100%;
            max-width: 700px;
        }
        .card {
            background-color: var(--card-background);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: var(--shadow);
            margin-bottom: 2rem;
        }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }
        h1 {
            text-align: center;
            color: var(--primary-color);
            margin-bottom: 2rem;
            flex: 1;
        }
        .theme-toggle {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border: 1px solid var(--border-color);
            background: var(--card-background);
            color: var(--text-color);
            border-radius: 999px;
            padding: 6px 10px;
            cursor: pointer;
            box-shadow: var(--shadow);
            user-select: none;
        }
        .theme-toggle input { display: none; }
        .theme-toggle .dot {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: var(--primary-color);
        }
        textarea {
            width: 100%;
            height: 160px;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            font-size: 1rem;
            font-family: 'Poppins', sans-serif;
            box-sizing: border-box;
            resize: vertical;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        textarea:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.2);
        }
        .input-actions {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            margin-top: 0.5rem;
        }
        .subtle-btn {
            border: 1px solid var(--border-color);
            background: transparent;
            color: var(--text-color);
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            cursor: pointer;
        }
        .counter { font-size: 0.85rem; opacity: 0.7; }
        button {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            padding: 1rem;
            border: none;
            border-radius: 8px;
            background: linear-gradient(90deg, #4a90e2, #50e3c2);
            color: white;
            font-size: 1.1rem;
            font-weight: 500;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 1rem;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
        }
        button:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
        }
        .result-card {
            display: {% if summary and sentiment %}block{% else %}none{% endif %};
        }
        .result-card h2 {
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 0.5rem;
            margin-top: 0;
        }
        .result-card h3 {
            color: var(--primary-color);
            font-size: 1rem;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        .result-card p {
            margin: 0;
            line-height: 1.6;
            word-wrap: break-word;
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .badge.positive { background: rgba(16,185,129,0.15); color: #10b981; }
        .badge.neutral { background: rgba(107,114,128,0.15); color: #9ca3af; }
        .badge.negative { background: rgba(239,68,68,0.15); color: #ef4444; }
        
        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top: 3px solid #fff;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Transcript Analysis AI ✨</h1>
            <label class="theme-toggle" title="Toggle dark mode">
                <input id="theme-switch" type="checkbox" />
                <span class="dot"></span>
            </label>
        </div>
        <div class="card">
            <form id="analysis-form" action="/analyze" method="post">
                <label for="transcript"><strong>Paste the call transcript below:</strong></label><br><br>
                <textarea id="transcript" name="transcript" required placeholder="e.g., Hi, I was trying to book a slot yesterday but the payment failed...">{{ request.form.get('transcript', '') }}</textarea>
                <div class="input-actions">
                    <div class="counter"><span id="char-count">0</span> characters</div>
                    <div style="display:flex; gap: 0.5rem;">
                        <button type="button" id="sample-btn" class="subtle-btn">Fill sample</button>
                        <button type="button" id="clear-btn" class="subtle-btn">Clear</button>
                    </div>
                </div>
                <button id="analyze-button" type="submit">Analyze Transcript</button>
            </form>
        </div>

        <div class="card result-card">
            <h2>Analysis Result</h2>
            {% if summary and sentiment %}
                <h3>Summary:</h3>
                <p id="summary-text">{{ summary }}</p>
                <div style="margin-top: 0.5rem; display:flex; gap:0.5rem;">
                    <button type="button" class="subtle-btn" data-copy-target="summary-text">Copy summary</button>
                </div>
                <h3>Sentiment:</h3>
                <p>
                    <span class="badge {% if sentiment.lower() == 'positive' %}positive{% elif sentiment.lower() == 'neutral' %}neutral{% elif sentiment.lower() == 'negative' %}negative{% endif %}">{{ sentiment }}</span>
                </p>
                <h3 style="margin-top: 2rem; font-size: 0.9rem; color: #777;">Data saved to {{ csv_file }}</h3>
            {% endif %}
        </div>
    </div>

    <script>
        const form = document.getElementById('analysis-form');
        const button = document.getElementById('analyze-button');
        const textarea = document.getElementById('transcript');
        const count = document.getElementById('char-count');
        const sampleBtn = document.getElementById('sample-btn');
        const clearBtn = document.getElementById('clear-btn');
        const themeSwitch = document.getElementById('theme-switch');

        form.addEventListener('submit', function() {
            button.disabled = true;
            button.innerHTML = '<div class="spinner"></div>Analyzing...';
        });

        // Character counter
        function updateCount() {
            count.textContent = textarea.value.length;
        }
        textarea.addEventListener('input', updateCount);
        updateCount();

        // Sample filler
        const sampleText = "Hello, I called yesterday about my order not arriving on time. The tracking shows it was delayed, and I was charged twice. The agent I spoke with was helpful, but the issue still isn't resolved. I would like a refund for the extra charge and confirmation on the new delivery date.";
        sampleBtn?.addEventListener('click', () => {
            textarea.value = sampleText;
            textarea.dispatchEvent(new Event('input'));
            textarea.focus();
        });
        clearBtn?.addEventListener('click', () => {
            textarea.value = '';
            textarea.dispatchEvent(new Event('input'));
            textarea.focus();
        });

        // Copy buttons
        document.querySelectorAll('[data-copy-target]')?.forEach(btn => {
            btn.addEventListener('click', async () => {
                const id = btn.getAttribute('data-copy-target');
                const el = document.getElementById(id);
                if (!el) return;
                try {
                    await navigator.clipboard.writeText(el.textContent || '');
                    btn.textContent = 'Copied!';
                    setTimeout(() => (btn.textContent = 'Copy summary'), 1200);
                } catch (e) {
                    console.error('Copy failed', e);
                }
            });
        });

        // Theme: read preference
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        const savedTheme = localStorage.getItem('theme');
        const initialDark = savedTheme ? savedTheme === 'dark' : prefersDark;
        document.documentElement.setAttribute('data-theme', initialDark ? 'dark' : 'light');
        themeSwitch.checked = initialDark;
        themeSwitch.addEventListener('change', () => {
            const theme = themeSwitch.checked ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('theme', theme);
        });
    </script>
</body>
</html>
"""

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    transcript = request.form.get('transcript')
    if not transcript:
        return "Error: No transcript provided.", 400
    summary, sentiment = analyze_transcript(transcript)
    print("--- Analysis Complete ---")
    print(f"Original Transcript: {transcript}")
    print(f"Summary: {summary}")
    print(f"Sentiment: {sentiment}")
    print("-------------------------")
    save_to_csv(transcript, summary, sentiment)
    return render_template_string(
        HTML_TEMPLATE, 
        transcript=transcript, 
        summary=summary, 
        sentiment=sentiment, 
        csv_file=CSV_FILE
    )

if __name__ == '__main__':
    app.run(debug=True)