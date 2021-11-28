import json
import time

from flask import Flask, render_template, request, jsonify
from run_for_flask import run
app = Flask(__name__)


@app.route('/', methods=['POST'])
def query():
    data = json.loads(request.get_data())
    run(
        realtime=False,
        reid=True,
        heatmap=True,
        yolo_weight="yolov5x.pt",
        reid_model="plr_osnet",
        deepsort_model="ckpt.t7",
        frame_skip=1,
        video_length=15,
        heatmap_accumulation=63,
        fps=15,
        videos_num=2,
        resolution='640'
    )
    time.sleep(2)
    return jsonify({"result" : data})


@app.route('/')
def hello_world():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
