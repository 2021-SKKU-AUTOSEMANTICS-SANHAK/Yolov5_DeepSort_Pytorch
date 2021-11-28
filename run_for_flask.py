import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from track import detect
import re_id as re
import frameget as fg
import queue
from multiprocessing import Process, Manager
import argparse
from yolov5.utils.general import check_img_size
import subprocess
import warnings
import torch
import time

def pstart(frame_get, frame_get2, count):
    if count != 0:
        cnt = 0

        while (cnt < count):
            p1 = Process(target=fg.get_frame, args=(0, frame_get), daemon=True)
            p2 = Process(target=fg.get_frame, args=(1, frame_get2), daemon=True)
            p1.start()
            p2.start()
            p1.join()
            p2.join()
            cnt += 1
    else:
        while True:
            p1 = Process(target=fg.get_frame, args=(0, frame_get), daemon=True)
            p2 = Process(target=fg.get_frame, args=(1, frame_get2), daemon=True)
            p1.start()
            p2.start()
            p1.join()
            p2.join()
            

def run(realtime, reid, heatmap, yolo_weight, reid_model, deepsort_model, frame_skip, video_length, heatmap_accumulation, fps, videos_num, resolution):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5l.pt', help='model.pt path')
    # parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
    #                     help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    # parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--modelpth', type=str, default='model_data/models/model.pth.tar-80', help='select reid model')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', default='0', type=int,
                        help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    # parser.add_argument("--realtime", type=int, default=0)
    parser.add_argument("--matrix", type=str, default='None')
    # parser.add_argument("--num_video", type=int, default=2)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--background", type=str, default='calliberation/background2.png')
    # parser.add_argument("--heatmap", type=str, default=1)
    # parser.add_argument("--frame", type=int, default=1)
    # parser.add_argument("--second", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=320)
    parser.add_argument("--video", type=str, default='None')
    # parser.add_argument("--heatmapsec", type=int, default=60)
    # parser.add_argument("--model", type=str, default='plr_osnet')
    # parser.add_argument("--fps", type=int, default=15)
    # parser.add_argument("--resolution", type=str, default='640')
    # parser.add_argument("--reid", type=str, default="on")

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    args.realtime = 1 if realtime else 0
    args.reid = "on" if reid else "off"
    args.heatmap = 1 if heatmap else 0
    args.save_vid = True
    args.frame = frame_skip
    args.second = video_length
    args.heatmapsec = heatmap_accumulation
    args.fps = fps
    args.num_video = videos_num
    args.resolution = resolution
    args.yolo_weights = "yolov5/weights/" + yolo_weight
    args.model = reid_model
    args.deep_sort_weights = "deep_sort_pytorch/deep_sort/deep/checkpoint/" + deepsort_model

    credential_path = "rapid-rite-331803-22fb7fcac271.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # reid = REID()
    # video1 = ['calliberation/in1_Trim.mp4']  # 엘리베이터
    # video2 = ['calliberation/in2_Trim.mp4']  # 입구
    # videos = [['calliberation/sample_video/ele.mp4'], ['calliberation/sample_video/en.mp4'], ['calliberation/sample_video/in.mp4']]  # 엘리베이터, 입구, 내부\
    str_video = ['cam1_daiso', 'cam2_daiso']
    videos = [['calliberation/cam1.mp4'], ['calliberation/cam2.mp4']]
    try:
        from torchreid.metrics.rank_cylib.rank_cy import evaluate_cy

        IS_CYTHON_AVAI = True
    except ImportError:
        IS_CYTHON_AVAI = False
        stdout = subprocess.run(['python ./torchreid/metrics/rank_cylib/setup.py build_ext --inplace'], shell=True)
        warnings.warn(
            'Cython does not work, will run cython'
        )

    with torch.no_grad():
        # mp.set_start_method('spawn')

        if args.realtime == 0:
            video1 = videos[0]
            video2 = videos[1]
            if args.video != 'None':
                x = args.video.split(',')
                video1 = videos[int(x[0])]
                if args.num_video == 2:
                    video2 = videos[int(x[1])]
                args.matrix = 'coor_' + str_video[int(x[0])]
                if args.num_video == 2:
                    args.matrix += ' coor_' + str_video[int(x[1])]
            now = time.localtime()
            date_time = time.strftime('%Y_%m_%d_%H_%M', now)
            args.date_time = date_time
            frame_get1 = queue.Queue()
            frame_get2 = queue.Queue()
            fg.get_frame_video(video1, frame_get1, args.frame, args.second, args.fps, args.resolution, args.limit)
            if args.num_video == 2:
                fg.get_frame_video(video2, frame_get2, args.frame, args.second, args.fps, args.resolution, args.limit)
            args.limit = frame_get1.qsize()
            if args.num_video == 2:
                size2 = frame_get2.qsize()
                if args.limit > size2:
                    args.limit = size2
            ids_per_frame1 = queue.Queue()
            ids_per_frame2 = queue.Queue()
            return_dict1 = queue.Queue()
            return_dict2 = queue.Queue()
            video_get1 = queue.Queue()
            video_get2 = queue.Queue()
            coor_get1 = queue.Queue()
            coor_get2 = queue.Queue()
            detect(args, frame_get1, return_dict1, ids_per_frame1, 'Video1', video_get1, coor_get1)
            if args.num_video == 2:
                detect(args, frame_get2, return_dict2, ids_per_frame2, 'Video2', video_get2, coor_get2)
            re.re_identification(args, return_dict1, return_dict2, ids_per_frame1, ids_per_frame2, video_get1, video_get2, coor_get1, coor_get2)

        else:
            now = time.localtime()
            date_time = time.strftime('%Y_%m_%d_%H_%M', now)
            args.date_time = date_time
            frame_get1 = Manager().Queue()
            frame_get2 = Manager().Queue()

            p0 = Process(target=pstart, args=(frame_get1, frame_get2, args.limit))
            p0.start()

            ids_per_frame1 = Manager().Queue()
            ids_per_frame2 = Manager().Queue()
            return_dict1 = Manager().Queue()
            return_dict2 = Manager().Queue()
            video_get1 = Manager().Queue()
            video_get2 = Manager().Queue()
            coor_get1 = Manager().Queue()
            coor_get2 = Manager().Queue()
            p5 = Process(target=detect,
                         args=(args, frame_get1, return_dict1, ids_per_frame1, 'Video1', video_get1, coor_get1),
                         daemon=True)
            if args.num_video == 2:
                p6 = Process(target=detect,
                             args=(args, frame_get2, return_dict2, ids_per_frame2, 'Video2', video_get2, coor_get2),
                             daemon=True)
            p7 = Process(target=re.re_identification,
                         args=(args, return_dict1, return_dict2, ids_per_frame1, ids_per_frame2,
                               video_get1, video_get2, coor_get1, coor_get2), daemon=True)
            p5.start()
            if args.num_video == 2:
                p6.start()
            p7.start()

            p7.join()

        # if args.realtime == 0:
        #     video1 = videos[0]
        #     video2 = videos[1]
        # if args.realtime == 0 and args.video != 'None':
        #     x = args.video.split(',')
        #     video1 = videos[int(x[0])]
        #     if args.num_video == 2:
        #         video2 = videos[int(x[1])]
        #     args.matrix = 'coor_' + str_video[int(x[0])]
        #     if args.num_video == 2:
        #         args.matrix += ' coor_' + str_video[int(x[1])]
        # now = time.localtime()
        # date_time = time.strftime('%Y_%m_%d_%H_%M', now)
        # args.date_time = date_time
        # frame_get1 = Manager().Queue()
        # frame_get2 = Manager().Queue()
        # if args.realtime:
        #     p0 = Process(target=pstart, args=(frame_get1, frame_get2, args.limit))
        #     p0.start()
        # else:
        #     p1 = Process(target=fg.get_frame_video,
        #                  args=(video1, frame_get1, args.frame, args.second, args.fps, args.resolution, args.limit))
        #     p1.start()
        #     if args.num_video == 2:
        #         p2 = Process(target=fg.get_frame_video,
        #                      args=(video2, frame_get2, args.frame, args.second, args.fps, args.resolution, args.limit))
        #         p2.start()
        #         p2.join()
        #     p1.join()
        #     args.limit = frame_get1.qsize()
        #     if args.num_video == 2:
        #         size2 = frame_get2.qsize()
        #         if args.limit > size2:
        #             args.limit = size2
        # # print(frame_get1.qsize())
        # # print(frame_get2.qsize())
        # ids_per_frame1 = Manager().Queue()
        # ids_per_frame2 = Manager().Queue()
        # return_dict1 = Manager().Queue()
        # return_dict2 = Manager().Queue()
        # video_get1 = Manager().Queue()
        # video_get2 = Manager().Queue()
        # coor_get1 = Manager().Queue()
        # coor_get2 = Manager().Queue()
        # p5 = Process(target=detect,
        #              args=(args, frame_get1, return_dict1, ids_per_frame1, 'Video1', video_get1, coor_get1),
        #              daemon=True)
        # if args.realtime == 1 or args.num_video == 2:
        #     p6 = Process(target=detect,
        #                  args=(args, frame_get2, return_dict2, ids_per_frame2, 'Video2', video_get2, coor_get2),
        #                  daemon=True)
        # p7 = Process(target=re.re_identification,
        #              args=(args, return_dict1, return_dict2, ids_per_frame1, ids_per_frame2,
        #                    video_get1, video_get2, coor_get1, coor_get2), daemon=True)
        # p5.start()
        # if args.num_video == 2:
        #     p6.start()
        # p7.start()
        #
        # p7.join()
# realtime, reid, heatmap, yolo_weight, reid_model, deepsort_model, frame_skip, video_length, heatmap_accumulation, fps, videos_num, resolution
# run(
#     realtime=False,
#     reid=True,
#     heatmap=True,
#     yolo_weight="yolov5x.pt",
#     reid_model="plr_osnet",
#     deepsort_model="ckpt.t7",
#     frame_skip=1,
#     video_length=15,
#     heatmap_accumulation=63,
#     fps=15,
#     videos_num=2,
#     resolution='640'
# )