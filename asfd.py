str_video = ['ele', 'en', 'in']
skip_frame_test = [1, 2, 3, 4, 5]
second_test = [15, 20, 30, 60, 120]
limit_test = [8, 6, 4, 2, 1]
resolution_test = ['640', '1280']
video_test = ['0,1', '1,2', '0,2']
test_output = open('output.txt', 'w')
test_output.write('{:5} {:5} {:4} {:3} {:5} {:5}\n'.format('vi1', 'vi2', 'skip', 'sec', 'resol', 'Time'))
for skip_test in skip_frame_test:
    for sec in range(len(second_test)):
        for resol_test in resolution_test:
            for vi_test in video_test:
                video = vi_test
                second = second_test[sec]
                limit = limit_test[sec]
                frame = skip_test
                resolution = resol_test
                test_output.write('{:5} {:5} {:4} {:3} {:5} {:.3f}\n'.format(vi_test[0], vi_test[2], frame, second, resolution, 12.45444))
test_output.close()