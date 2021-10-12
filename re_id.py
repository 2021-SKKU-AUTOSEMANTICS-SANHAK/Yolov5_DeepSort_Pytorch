from reid import REID
import time
import queue as Queue
import numpy as np
import Heatmap as ht
import copy
import operator

def re_identification(args, return_dict1, return_dict2, ids_per_frame1_list, ids_per_frame2_list, video_get1, video_get2, coor_get1, coor_get2):
    reid = REID(args)
    num_video = args.num_video
    thres = int(args.heatmapsec / args.second)
    if args.matrix == 'None':
        M2 = np.load("calliberation/coor_en_640.npy")
        M2 = np.array(M2, np.float32)
        f2 = open('calliberation/coor_en_640.txt', 'r')
        line2 = f2.readline()
        coor2 = line2.split(' ')
        f2.close()
        M1 = np.load("calliberation/coor_ele_640.npy")
        M1 = np.array(M1, np.float32)
        f1 = open('calliberation/coor_ele_640.txt', 'r')
        line1 = f1.readline()
        coor1 = line1.split(' ')
        f1.close()
    else:
        x = args.matrix.split(' ')
        print(x)
        M1 = np.load("calliberation/"+x[0] + ".npy")
        M1 = np.array(M1, np.float32)
        f1 = open("calliberation/"+x[0] + ".txt", 'r')
        line1 = f1.readline()
        coor1 = line1.split(' ')
        f1.close()
        if len(x) == 2 and num_video == 2:
            M2 = np.load("calliberation/" + x[1] + ".npy")
            M2 = np.array(M2, np.float32)
            f2 = open("calliberation/" + x[1] + ".txt", 'r')
            line2 = f2.readline()
            coor2 = line2.split(' ')
            f2.close()
        else:
            M2 = np.load("calliberation/coor_en_640.npy")
            M2 = np.array(M2, np.float32)
            f2 = open('calliberation/coor_en_640.txt', 'r')
            line2 = f2.readline()
            coor2 = line2.split(' ')
            f2.close()

    count = 0
    heatmapcount = 0
    example_points = []
    heat_name = 0
    while True:
        if args.realtime == 1 or num_video == 2:
            while (return_dict1.empty()) or (return_dict2.empty()) or (ids_per_frame1_list.empty())\
                    or ids_per_frame2_list.empty() or coor_get1.empty() or coor_get2.empty():
                    time.sleep(1)
            start_time = time.time()
            return_list = return_dict1.get()
            return_list2 = return_dict2.get()

            ids_per_frame1 = ids_per_frame1_list.get()
            ids_per_frame2 = ids_per_frame2_list.get()
            threshold = args.threshold
            exist_ids = set()
            final_fuse_id = dict()
            ids_per_frame22 = []
            feats = dict()
            size = len(return_list)
            #print(size)
            """
            print('video1')
            for key, value in return_list.items():
                print(key)
            print('video2')
            """
            for key, value in return_list2.items():
                return_list[key + size] = return_list2[key]
                #print('{}, {}'.format(key, key+size))
            images_by_id = copy.deepcopy(return_list)
            #print('done')
            """
            for key, value in images_by_id.items():
                print(key)
            """
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
                feats[i] = reid._features(images_by_id[i][:min(len(images_by_id[i]), 60)])
            reid_dict = dict()
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
                            if len(images_by_id[nid]) < 5:
                                exist_ids.add(nid)
                                continue
                            unpickable = []
                            for i in f:
                                for key, item in final_fuse_id.items():
                                    if i in item:
                                        unpickable += final_fuse_id[key]
                            #print('exist_ids {} unpickable {}'.format(exist_ids, unpickable))
                            for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                                tmp = np.mean(reid.compute_distance(feats[nid], feats[oid]))

                                #print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                                dis.append([oid, tmp])
                            exist_ids.add(nid)
                            if not dis:
                                final_fuse_id[nid] = [nid]
                                continue
                            dis.sort(key=operator.itemgetter(1))
                            if dis[0][1] < threshold:
                                combined_id = dis[0][0]
                                #print(dis[0][1])
                                print('oid {} , nid {} , tmp {}'.format(combined_id, nid, dis[0][1]))
                                images_by_id[combined_id] += images_by_id[nid]
                                final_fuse_id[combined_id].append(nid)
                                reid_dict[nid] = combined_id
                            else:
                                final_fuse_id[nid] = [nid]

            print('people : {}. ID : {}'.format(len(final_fuse_id), final_fuse_id))
            heatmapcount += 1
            heatmapcount = heatmapcount % thres
            if count+1 == args.limit and args.realtime != 1:
                heatmapcount = 0

            ht.store(video_get1, video_get2, size, coor_get1, coor_get2, M1, M2, coor1, coor2,
                     count, 2, final_fuse_id, reid_dict, args.background, args.save_vid,
                     args.save_txt, heatmapcount, example_points, heat_name)
            if heatmapcount == 0:
                example_points = []
                heat_name += 1

        else:
            final_fuse_id = dict()
            reid_dict = dict()
            heatmapcount += 1
            heatmapcount = heatmapcount % thres
            if count + 1 == args.limit and args.realtime != 1:
                heatmapcount = 0

            ht.store(video_get1, video_get2, 0, coor_get1, coor_get2, M1, M2, coor1, coor2,
                     count, 1, final_fuse_id, reid_dict, args.background, args.save_vid,
                     args.save_txt, heatmapcount, example_points, heat_name)

            if heatmapcount == 0:
                example_points = []
                heat_name += 1
        #print(reid_dict)

        count += 1
        if args.realtime != 1 and count == args.limit:
            break
        elif args.limit != 0 and count == args.limit:
            break
