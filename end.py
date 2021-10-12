import sys

sys.path.insert(0, './yolov5')
from PIL import Image
import torch.multiprocessing as mp
import warnings
from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from multiprocessing import Process, Manager
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from reid import REID
import people_counting_v2
import collections
import copy
import numpy as np
import operator
import cv2
import multiprocessing as mp
import queue as Queue
from itertools import chain
from google.cloud import bigquery, storage
import multiprocessing


def get_frame(i, frame):
    project_id = 'atsm-202107'
    bucket_id = 'sanhak_2021'
    dataset_id = 'sanhak_2021'
    table_id = 'video_sec-10_frame-4'

    storage_client = storage.Client()
    db_client = bigquery.Client()
    bucket = storage_client.bucket(bucket_id)
    select_query = (
        "SELECT camID, date_time, path FROM `{}.{}.{}` WHERE camID = {} ORDER BY date_time LIMIT 1".format(project_id,
                                                                                                           dataset_id,
                                                                                                           table_id, i))
    query_job = db_client.query(select_query)
    results = query_job.result()
    for row in results:
        path = row.path
        dt = row.date_time

    delete_query = (
        "DELETE FROM `{}.{}.{}` WHERE date_time = '{}' AND camID = {}".format(project_id, dataset_id, table_id, dt, i))

    query_job = db_client.query(delete_query)
    results = query_job.result()
    save = []
    cam = cv2.VideoCapture(path)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if cam.isOpened():
        while True:
            ret, img = cam.read()
            if ret:
                cv2.waitKey(33)  # what is this??
                save.append(img)
            else:
                break
        frame.put(save)





    else:
        print('cannot open the vid #' + str(i))
        exit()
    # while True:
    #     ret, realframe = cam.read()
    #     if (time.time() - start_time) >= 3:
    #         cam.release()
    #         break
    #     frame.append(realframe)



def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)





def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def detect(opt, dataset_list, return_dict, ids_per_frame_list, string, video_get):
    out, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.img_size, opt.evaluate
    time_init = time.time()
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    # Initialize
    device = select_device(opt.device)
    """
    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    """
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    while True:
        print(string + 'start')
        while (dataset_list.empty()):
            time.sleep(1)
        start_time = time.time()
        dataset = dataset_list.get()
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        track_cnt = dict()
        # print('time (init) : {}'.format(time.time() - time_init))
        t0 = time.time()
        frame_cnt = 1
        images_by_id = dict()
        ids_per_frame = []
        drawimage=[]
        for im0s in dataset:
            img = letterbox(im0s, 640, stride=32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3 x 416 x 416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                # if webcam:  # batch_size >= 1
                #    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                # else:
                s, im0 = '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string
                # save_path = str(Path(out) / Path(p).name)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    xywh_bboxs = []
                    confs = []

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        # to deep sort format
                        x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                        xywh_bboxs.append(xywh_obj)
                        confs.append([conf.item()])

                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confs)

                    # pass detections to deepsort
                    outputs, images_by_id = deepsort.update(xywhs, confss, im0, images_by_id, ids_per_frame, track_cnt,
                                                            frame_cnt)


                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)
                else:
                    deepsort.increment_ages()
                # Print time (inference + NMS)
                print('{}, {}/{} {}Done. ({}s)'.format(string, frame_cnt, len(dataset), s, t2 - t1))

            drawimage.append(im0)
            frame_cnt += 1

        video_get.put(drawimage)
        video_get.put(track_cnt)
        return_dict.put(images_by_id)
        ids_per_frame_list.put(ids_per_frame)
        print(string + ' Tracking Done')


def re_identification(return_dict1, return_dict2, ids_per_frame1_list, ids_per_frame2_list, video_get1, video_get2):
    reid = REID()
    count = 0
    while True:
        while (return_dict1.empty()) or (return_dict2.empty()) or (ids_per_frame1_list.empty()) or ids_per_frame2_list.empty():
                time.sleep(1)
        start_time = time.time()
        return_list = return_dict1.get()
        return_list2 = return_dict2.get()


        ids_per_frame1 = ids_per_frame1_list.get()
        ids_per_frame2 = ids_per_frame2_list.get()
        threshold = 320
        exist_ids = set()
        final_fuse_id = dict()
        ids_per_frame = []
        ids_per_frame22 = []
        images_by_id = dict()
        feats = dict()
        size = len(return_list)
        for key, value in return_list2.items():
            return_list[key + size] = return_list2[key]
        images_by_id = copy.deepcopy(return_list)
        print(len(images_by_id))

        for i in ids_per_frame2:
            d = set()
            for k in i:
                k += size
                d.add(k)
            ids_per_frame22.append(d)

        ids_per_frame = copy.deepcopy(ids_per_frame1)
        for k in ids_per_frame22:
            ids_per_frame.append(k)

        for i in images_by_id:
            feats[i] = reid._features(images_by_id[i])

        for f in ids_per_frame:
            if f:
                if len(exist_ids) == 0:
                    for i in f:
                        final_fuse_id[i] = [i]
                    exist_ids = exist_ids or f
                else:
                    new_ids = f - exist_ids
                    for nid in new_ids:
                        dis = []
                        if len(images_by_id[nid]) < 10:
                            exist_ids.add(nid)
                            continue
                        unpickable = []
                        for i in f:
                            for key, item in final_fuse_id.items():
                                if i in item:
                                    unpickable += final_fuse_id[key]
                        print('exist_ids {} unpickable {}'.format(exist_ids, unpickable))
                        for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                            tmp = np.mean(reid.compute_distance(feats[nid], feats[oid]))
                            print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                            dis.append([oid, tmp])
                        exist_ids.add(nid)
                        if not dis:
                            final_fuse_id[nid] = [nid]
                            continue
                        dis.sort(key=operator.itemgetter(1))
                        if dis[0][1] < threshold:
                            combined_id = dis[0][0]
                            images_by_id[combined_id] += images_by_id[nid]
                            final_fuse_id[combined_id].append(nid)
                        else:
                            final_fuse_id[nid] = [nid]

        print('Final ids and their sub-ids:', final_fuse_id)
        print('people : ', len(final_fuse_id))
        heatmapmake1 = dict()
        heatmapmake2 = dict()

        drawimage = video_get1.get()  # list
        size2 = len(drawimage)
        track_cnt1 = video_get1.get()  # dict plus id 해야됨
        imag2 = video_get2.get()
        track_cnt2 = video_get2.get()
        for a in imag2:
            drawimage.append(a)
        for key, value in track_cnt2.items():
            for a in range(len(track_cnt2[key])):
                track_cnt2[key][a][0] +=size2
            track_cnt1[key + size] = track_cnt2[key]

        output = str(count) + '.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output, fourcc, 7.5, (640, 480))
        check = set(list())
        for frame in range(len(drawimage)):
           img = drawimage[frame]
           for idx in final_fuse_id:
             for i in final_fuse_id[idx]: #i = id
                for f in track_cnt1[i]:
                    if frame == f[0]:
                        text_scale, text_thickness, line_thickness = get_FrameLabels(img)
                        cv2_addBox(idx, img, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
           print(img)
           out.write(img)
        out.release()
        count = count + 1




def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness

def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
    cv2.putText(frame, str(track_id), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                thickness=text_thickness)
def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color







warnings.filterwarnings('ignore')


def pstart(frame_get, frame_get2):
    cnt = 0



    p1 = Process(target=get_frame, args=(0, frame_get))
    p2 = Process(target=get_frame, args=(1, frame_get2))
    p1.start()
    p2.start()
    p1.join()
    p2.join()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/models/yolov5x.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
                        help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, default='0')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()

    credential_path = "atsm-202107-50b0c3dc3869.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    multiprocessing.set_start_method("spawn")
    frame_get1 = Manager().Queue()
    frame_get2 = Manager().Queue()
    args.img_size = check_img_size(args.img_size)
    p0 = Process(target=pstart, args=(frame_get1, frame_get2))
    p0.start()

    with torch.no_grad():
        ids_per_frame1 = Manager().Queue()
        ids_per_frame2 = Manager().Queue()
        return_dict1 = Manager().Queue()
        return_dict2 = Manager().Queue()
        video_get1 = Manager().Queue()
        video_get2 = Manager().Queue()
        p5 = mp.Process(target=detect, args=(args, frame_get1, return_dict1, ids_per_frame1, 'Video1', video_get1))
        p6 = mp.Process(target=detect, args=(args, frame_get2, return_dict2, ids_per_frame2, 'Video2', video_get2))
        p7 = mp.Process(target=re_identification, args=(return_dict1, return_dict2, ids_per_frame1, ids_per_frame2, video_get1, video_get2))
        p5.start()
        p6.start()
        p7.start()
        p5.join()
        p6.join()
        p7.join()
        while (1):
            time.sleep(1)

