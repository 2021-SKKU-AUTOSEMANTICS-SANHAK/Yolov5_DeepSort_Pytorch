import json
import time

from flask import Flask, render_template, request, jsonify, redirect, url_for
from run_for_flask import run
app = Flask(__name__)
app.config.update(
    DEBUG = True,
    TEMPLATES_AUTO_RELOAD = True,
    SEND_FILE_MAX_AGE_DEFAULT = 0
)

file_name = None
first_line = None
second_line = None


@app.route('/outputs', methods=['GET', 'POST'])
def outputs():
    return render_template("outputs.html", pref=file_name, first_line=first_line, second_line=second_line)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    global file_name
    global first_line
    global second_line
    if request.method == 'GET':
        return render_template("index.html", output={'datetime' : 'DoNotDelete'})
    if request.method == 'POST':
        data = json.loads(request.get_data())
        date_time = run(
            realtime=data['realtime'],
            reid=data['reid'],
            heatmap=data['heatmap'],
            yolo_weight=data['yolo_weight'],
            reid_model=data['reid_model'],
            deepsort_model=data['deepsort_model'],
            frame_skip=data['frame_skip'],
            video_length=data['video_length'],
            heatmap_accumulation=data['heatmap_accumulation'],
            fps=data['fps'],
            videos_num=data['videos_num'],
            resolution=data['resolution']
        )
        time.sleep(1)
        file_name = date_time

        with open("static/cortxt/" + date_time + ".txt") as f:
            first_line = f.readline()
            second_line = f.readline()

        return jsonify({'datetime': file_name})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
