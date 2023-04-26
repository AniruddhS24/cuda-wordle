from flask import Flask, request
from flask_cors import CORS
from subprocess import Popen, PIPE
import json
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    # serve index.html
    return app.send_static_file('index.html')

@app.route('/query', methods=['POST'])
def query_solver():
    data = request.get_json()
    guesses = data['guesses']
    colors = data['colors']
    command = ["./interactive_solver", "-d", "./basic_dictionary/potential_words.txt", "-v", "./basic_dictionary/vocab.txt", "--guesses", f"{guesses}", "--colors",  f"{colors}"]
    print(command)
    process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode('utf-8')
    lines = stdout.split('\n')
    packet = {}
    for line in lines:
        if 'Dictionary Size' in line:
            packet['dictionary_size'] = int(line.split(' ')[-1])
    idx = lines.index('DISTRIBUTION:') + 1
    packet['distribution'] = lines[idx:]
    print(packet['distribution'])
    return json.dumps(packet)

@app.route('/alive', methods=['GET'])
def alive():
    return 'I am alive!'


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(debug=False, host='0.0.0.0', port=port)