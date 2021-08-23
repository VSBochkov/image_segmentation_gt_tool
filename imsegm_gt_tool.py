import numpy as np
import cv2
import os
import sys
import copy
import time

imh, imw = 0, 0
origin_img = None
overlay = None
color_near_dist = 40.

NO_ACTION = 0
CREATE_RECTANGLE = 1
FILL_NEAR_PIX = 2
UNFILL_NEAR_PIX = 3
FILL_ALL_PIX = 4

BUT_RIGHT = 83
BUT_LEFT = 81
BUT_BACKSPACE = 8
BUT_PLUS = 43
BUT_MINUS = 45
BUT_ENTER = 13
BUT_ESC = 27

sys.setrecursionlimit(1000000)
Z_WERE_PRESSED = False


def keep_data(dict):
    global settings
    return {
        'class_val': dict['class_val'],
        'class_val_to_rect': {__class['class_id']: copy.deepcopy(dict['class_val_to_rect'][__class['class_id']]) for __class in settings['classes']},
        'class_val_to_dist': {__class['class_id']: copy.deepcopy(dict['class_val_to_dist'][__class['class_id']]) for __class in settings['classes']},
        'labeled_mask': dict['labeled_mask'].copy()
    }


# recursive
def fill_near_similar_pix(j, i, color, class_val):
    global overlay
    difference = np.linalg.norm(np.asarray(origin_img[i, j], dtype=float) - np.asarray(color, dtype=float))
    if difference <= curr_dict['class_val_to_dist'][class_val]:
        curr_dict['labeled_mask'][i, j] = class_val
        p1_x, p1_y = curr_dict['class_val_to_rect'][class_val][0]
        p2_x, p2_y = curr_dict['class_val_to_rect'][class_val][1]
        neighb = [(i, j - 1), (i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j + 1), (i + 1, j + 1), (i + 1, j), (i + 1, j - 1)]
        for i, j in [(i, j) for i, j in neighb if p1_y <= i < p2_y and p1_x <= j < p2_x and curr_dict['labeled_mask'][i, j] == 0]:
            fill_near_similar_pix(j, i, color, class_val)


# recursive
def unfill_near_similar_pix(j, i, color, class_val):
    global curr_dict
    difference = np.linalg.norm(np.asarray(origin_img[i, j], dtype=float) - np.asarray(color, dtype=float))
    if difference <= curr_dict['class_val_to_dist'][class_val]:
        curr_dict['labeled_mask'][i, j] = 0
        p1_x, p1_y = curr_dict['class_val_to_rect'][class_val][0]
        p2_x, p2_y = curr_dict['class_val_to_rect'][class_val][1]
        neighb = [(i, j - 1), (i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j + 1), (i + 1, j + 1), (i + 1, j), (i + 1, j - 1)]
        for i, j in [(i, j) for i, j in neighb if p1_y <= i < p2_y and p1_x <= j < p2_x and curr_dict['labeled_mask'][i, j] == class_val]:
            unfill_near_similar_pix(j, i, color, class_val)


# non-recursive
def fill_all_similar_pix(color, class_val):
    global origin_img, curr_dict
    difference = np.linalg.norm(np.asarray(origin_img, dtype=float) - np.asarray(color, dtype=float), axis=2)
    difference[difference <= curr_dict['class_val_to_dist'][class_val]] = class_val
    difference[difference > curr_dict['class_val_to_dist'][class_val]] = 0.
    difference = np.asarray(difference, dtype=int)
    p1_x, p1_y = curr_dict['class_val_to_rect'][class_val][0]
    p2_x, p2_y = curr_dict['class_val_to_rect'][class_val][1]
    diff_bounded = np.zeros_like(difference)
    diff_bounded[p1_y: p2_y, p1_x: p2_x] = difference[p1_y: p2_y, p1_x: p2_x].copy()
    for i in range(p1_y, p2_y):
        for j in range(p1_x, p2_x):
            if curr_dict['labeled_mask'][i, j] != 0:
                diff_bounded[i, j] = 0
    difference = diff_bounded.copy()
    curr_dict['labeled_mask'] = curr_dict['labeled_mask'] + difference


def allign_to_frame_bounds(pt1_x, pt2_x, pt1_y, pt2_y):
    if pt1_x <= 10:
        pt1_x = 0
    if pt1_x >= imw - 10:
        pt1_x = imw
    if pt2_x <= 10:
        pt2_x = 0
    if pt2_x >= imw - 10:
        pt2_x = imw
    if pt1_y <= 10:
        pt1_y = 0
    if pt1_y >= imh - 10:
        pt1_y = imh
    if pt2_y <= 10:
        pt2_y = 0
    if pt2_y >= imh - 10:
        pt2_y = imh
    return pt1_x, pt2_x, pt1_y, pt2_y


def draw_selected_area_where_fire(class_val):
    global curr_dict, overlay, imh, imw
    pt1, pt2 = curr_dict['class_val_to_rect'][class_val][0], curr_dict['class_val_to_rect'][class_val][1]
    overlay[0: imh, 0: pt1[0]] = overlay[0: imh, 0: pt1[0]] / 2
    overlay[0: pt1[1], pt1[0]: pt2[0]] = overlay[0: pt1[1], pt1[0]: pt2[0]] / 2
    overlay[0: imh, pt2[0]: imw] = overlay[0: imh, pt2[0]: imw] / 2
    overlay[pt2[1]: imh, pt1[0]: pt2[0]] = overlay[pt2[1]: imh, pt1[0]: pt2[0]] / 2
    cv2.rectangle(overlay, pt1, pt2, class_val_to_lblcolor[class_val], 1)


last_x, last_y, last_event = None, None, None


def on_mouse_click(event, x, y, flags, _):
    global select_classes, select_classes_hints, prev_dict, overlay, curr_dict, class_val_to_str, origin_img, Z_WERE_PRESSED
    global last_x, last_y, last_event
    global but, BUT_MINUS, BUT_PLUS
    if curr_dict['class_val'] is None or event is None:
        return
    if event == cv2.EVENT_LBUTTONUP:
        if Z_WERE_PRESSED:
            if len(class_val_to_rect[curr_dict['class_val']]) < 2:
                class_val_to_rect[curr_dict['class_val']].append((x, y))
            if len(class_val_to_rect[curr_dict['class_val']]) == 2:
                Z_WERE_PRESSED = False
                pt1_x, pt1_y = class_val_to_rect[curr_dict['class_val']][0]
                pt2_x, pt2_y = class_val_to_rect[curr_dict['class_val']][1]
                pt1_x, pt2_x, pt1_y, pt2_y = allign_to_frame_bounds(pt1_x, pt2_x, pt1_y, pt2_y)
                curr_dict['class_val_to_rect'][curr_dict['class_val']] = [(min(pt1_x, pt2_x), min(pt1_y, pt2_y)), (max(pt1_x, pt2_x), max(pt1_y, pt2_y))]
                draw(curr_dict['class_val'])
                select_classes = copy.deepcopy(select_classes_hints)
                select_classes.extend(actions_with_classes_hints)
                draw_status('MARKUP "{}" CLASS PIXELS'.format(class_val_to_str[curr_dict['class_val']]), select_classes)
                prev_dict = keep_data(curr_dict)
        else:
            print('LBUTTON_UP X = {}, Y = {}, class = {}, dist = {}'.format(
                x, y, curr_dict['class_val'], curr_dict['class_val_to_dist'][curr_dict['class_val']]))
            if but != BUT_PLUS and but != BUT_MINUS:
                print('but != BUT_PLUS and but != BUT_MINUS')
                prev_dict = keep_data(curr_dict)
            if curr_dict['labeled_mask'][y, x] == curr_dict['class_val']:
                print('unfill_near_similar_pix')
                unfill_near_similar_pix(x, y, origin_img[y, x], curr_dict['class_val'])
            else:
                print('fill_near_similar_pix')
                fill_near_similar_pix(x, y, origin_img[y, x], curr_dict['class_val'])
            draw(curr_dict['class_val'])
            select_classes = copy.deepcopy(select_classes_hints)
            select_classes.extend(actions_with_classes_hints)
            draw_status('MARKUP "{}" CLASS PIXELS'.format(class_val_to_str[curr_dict['class_val']]), select_classes)
        last_x, last_y, last_event = x, y, event
    elif event == cv2.EVENT_MBUTTONUP:
        print('MBUTTON_UP X = {}, Y = {}, class = {}, dist = {}'.format(
            x, y, curr_dict['class_val'], curr_dict['class_val_to_dist'][curr_dict['class_val']]))
        if but != BUT_PLUS and but != BUT_MINUS:
            print('but != BUT_PLUS and but != BUT_MINUS')
            prev_dict = keep_data(curr_dict)
        fill_all_similar_pix(origin_img[y, x], curr_dict['class_val'])
        draw(curr_dict['class_val'])
        select_classes = copy.deepcopy(select_classes_hints)
        select_classes.extend(actions_with_classes_hints)
        draw_status('MARKUP "{}" CLASS PIXELS'.format(class_val_to_str[curr_dict['class_val']]), select_classes)
        last_x, last_y, last_event = x, y, event


def get_average_color(col1, col2):
    return np.asarray(
        (np.asarray(col1, dtype=int) + np.asarray(col2, dtype=int)) / 2,
        dtype=np.uint8
    )


def draw_aver_masks(class_val):
    global classes, overlay, curr_dict
    for i in range(0, curr_dict['labeled_mask'].shape[0]):
        for j in range(0, curr_dict['labeled_mask'].shape[1]):
            if curr_dict['labeled_mask'][i, j] == class_val:
                overlay[i, j] = class_val_to_lblcolor[class_val]
            elif curr_dict['labeled_mask'][i, j] > 0:
                overlay[i, j] = get_average_color(class_val_to_lblcolor[curr_dict['labeled_mask'][i, j]], origin_img[i, j])
            else:
                overlay[i, j] = origin_img[i, j]


def get_mask_img():
    labeled_mask_img = np.zeros((curr_dict['labeled_mask'].shape[0], curr_dict['labeled_mask'].shape[1], 3), dtype=np.uint8)
    for i in range(0, curr_dict['labeled_mask'].shape[0]):
        for j in range(0, curr_dict['labeled_mask'].shape[1]):
            labeled_mask_img[i, j] = class_val_to_lblcolor[curr_dict['labeled_mask'][i, j]] if curr_dict['labeled_mask'][i, j] > 0 else np.zeros(3)
    return labeled_mask_img


show_masks = False
origin_imgs = {}


def open_and_read(video_path, frame_num):
    global video_name, origin_imgs

    cap = cv2.VideoCapture(video_path)
    while not cap.isOpened():
        cap.release()
        cap.open(video_path)
        time.sleep(1)

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    imh, imw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # Define the codec and create VideoWriter object

    for i in range(0, frame_num + int(fps)):
        ok, origin_img = cap.read()
        if not ok:
            break
        if i > frame_num - fps:
            origin_imgs[i] = origin_img

    cap.release()


def get_frame(video_path, number):
    global origin_imgs
    if len(origin_imgs) == 0 or number > max(origin_imgs.keys()):
        open_and_read(video_path, number)
    return origin_imgs[number]


def get_overlay():
    global curr_dict, overlay
    for i in range(0, imh):
        for j in range(0, imw):
            if curr_dict['labeled_mask'][i, j] > 0:
                overlay[i, j] = class_val_to_lblcolor[curr_dict['labeled_mask'][i, j]]
            else:
                overlay[i, j] = origin_img[i, j]


def draw(class_val=None):
    global overlay, curr_dict
    if class_val is None:
        get_overlay()
    else:
        draw_aver_masks(class_val)
        draw_selected_area_where_fire(class_val)
    cv2.imshow('labeled_mask', get_mask_img())
    cv2.imshow('overlay', overlay)


def draw_status(status, commands):
    status_img = np.zeros((600, 600, 3), dtype=np.uint8)
    cv2.putText(status_img, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0xff, 0xff, 0xff))
    for i, command in enumerate(commands):
        cv2.putText(status_img, command, (10, (i + 2) * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0xff, 0xff))
    cv2.imshow('Status', status_img)


if __name__ == '__main__':
    import json

    settings = json.load(open('config.json', 'r'))
    print(settings)

    class_val_to_str = {__class['class_id']: __class['class_name'] for __class in settings['classes']}
    class_val_to_lblcolor = {__class['class_id']: __class['marked_up_color'] for __class in settings['classes']}
    class_val_to_rect = {__class['class_id']: [] for __class in settings['classes']}
    class_val_to_dist = {__class['class_id']: __class['near_color_rad'] for __class in settings['classes']}

    classes_buttons = {ord('{}'.format(i)) : i for i in range(1, len(settings['classes']) + 1)}
    control_buttons = [ord('z'), BUT_RIGHT, BUT_LEFT, BUT_BACKSPACE, BUT_PLUS, BUT_MINUS, BUT_ENTER, BUT_ESC]

    select_classes_hints = ["'{}' - SELECT '{}'".format(i, class_val_to_str[i]) for i in range(1, len(settings['classes']) + 1)]
    actions_with_classes_hints = ["'z' SET A RECTANGLE", "'backspace' REVERT ACTION",
                                  "'+' INCREASE VALUE OF SIMILAR COLOR",
                                  "'-' INCREASE VALUE OF SIMILAR COLOR",
                                  "'enter' SAVE LABELED MASK AND FINISH",
                                  "'esc' exit"]

    curr_dict = {
        'class_val': None,
        'class_val_to_rect': {__class['class_id']: [] for __class in settings['classes']},
        'class_val_to_dist': {__class['class_id']: __class['near_color_rad'] for __class in settings['classes']},
        'labeled_mask': None
    }
    prev_dict = curr_dict

    gt_labels_path = os.path.join(settings['output_labeled_data_folder'], 'labels')
    if not os.path.exists(gt_labels_path):
        os.makedirs(gt_labels_path)
    gt_images_path = os.path.join(settings['output_labeled_data_folder'], 'images')
    if not os.path.exists(gt_images_path):
        os.makedirs(gt_images_path)

    out_image_path = ''
    out_labeled_mask_path = ''
    out_labeled_overlay_path = ''

    cv2.namedWindow('labeled_mask')
    cv2.namedWindow('overlay')
    cv2.namedWindow('Status')
    cv2.setMouseCallback('overlay', on_mouse_click)
    imh, imw = 0, 0
    if 'image_path' in settings:
        USE_IMAGE = True
        origin_img = cv2.imread(settings['image_path'])
        overlay = origin_img.copy()
        imh, imw = origin_img.shape[0], origin_img.shape[1]
        image_name_ext = settings['image_path'][settings['image_path'].rfind('/') + 1 :]
        out_image_path = os.path.join(gt_images_path, image_name_ext)
        image_name = image_name_ext[: settings['image_path'].rfind('.')]
        out_labeled_mask_path = os.path.join(gt_labels_path, '{}_mask.npy'.format(image_name))
        if os.path.exists(out_labeled_mask_path):
            # Load previously marked up data
            curr_dict['labeled_mask'] = np.load(out_labeled_mask_path)
        else:
            curr_dict['labeled_mask'] = np.zeros((imh, imw), dtype=np.uint8)
        out_labeled_overlay_path = os.path.join(gt_labels_path, '{}_overlay.jpeg'.format(image_name))
    elif 'video_path' in settings:
        USE_IMAGE = False
        cap = cv2.VideoCapture(settings['video_path'])
        while not cap.isOpened():
            cap.release()
            cap.open(settings['video_path'])
            time.sleep(1)

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        imh, imw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        # Define the codec and create VideoWriter object
        print('resolution = [h: {}, w: {}], number of frames = {}'.format(imh, imw, frames))

        frame_num = 0
        if 'video_time' in settings:
            time_min = int(settings['video_time'].split(':')[0])
            time_sec = int(settings['video_time'].split(':')[1])
            frame_num = int((time_min * 60 + time_sec) * fps)
        elif 'frame_num' in settings:
            frame_num = int(settings['frame_num'])
        else:
            print('You must specify video_time or frame_num parameter in config.json when video_path option is used')
            exit(-1)

        origin_img = get_frame(settings['video_path'], frame_num)
        overlay = origin_img.copy()

        draw_status('IMAGE_NOT_SELECTED', ['-> NEXT FRAME', '<- PREVIOUS FRAME', "'enter' take image into markup", "'esc' exit"])

        video_name = settings['video_path'][settings['video_path'].rfind('/') + 1 : settings['video_path'].rfind('.')]

        IS_IMAGE_CHOSEN = False
        while not IS_IMAGE_CHOSEN:
            out_image_path = os.path.join(gt_images_path, '{}_frame#{}.jpeg'.format(video_name, frame_num))
            out_labeled_mask_path = os.path.join(gt_labels_path, '{}_frame#{}_mask.npy'.format(video_name, frame_num))
            out_labeled_overlay_path = os.path.join(gt_labels_path, '{}_frame#{}_overlay.jpeg'.format(video_name, frame_num))
            if os.path.exists(out_labeled_mask_path):
                # Load previously marked up data
                curr_dict['labeled_mask'] = np.load(out_labeled_mask_path)
            else:
                curr_dict['labeled_mask'] = np.zeros((imh, imw), dtype=np.uint8)
            draw()
            but = cv2.waitKey(0)
            print('but is {}'.format(but))
            if but in control_buttons:
                if but == BUT_RIGHT:
                    frame_num += 1
                elif but == BUT_LEFT:
                    frame_num -= 1
                elif but == BUT_ENTER:
                    IS_IMAGE_CHOSEN = True
                elif but == BUT_ESC:
                    exit(0)
                origin_img = get_frame(settings['video_path'], frame_num)
    else:
        print('You must specify either image_path or video_path key in config.json')
        exit(-1)

    select_classes = copy.deepcopy(select_classes_hints)
    select_classes.extend(["'enter' SAVE LABELED MASK AND FINISH"])
    draw_status('IMAGE_SELECTED', select_classes)
    but = -1
    prev_but = but
    __prev_dict = {}
    if os.path.exists(out_labeled_mask_path):
        curr_dict['labeled_mask'] = np.load(out_labeled_mask_path)
        draw()
    while but != BUT_ENTER:
        if curr_dict['class_val'] is not None and len(curr_dict['class_val_to_rect'][curr_dict['class_val']]) == 2:
            draw(curr_dict['class_val'])
        else:
            draw()
        but = cv2.waitKey(0)
        if but in control_buttons:
            if but == ord('z'):
                Z_WERE_PRESSED = not Z_WERE_PRESSED
                class_val_to_rect[curr_dict['class_val']].clear()
                draw_status('CREATING RECTANGLE FOR "{}"'.format(class_val_to_str[curr_dict['class_val']]),
                            ["click in overlay window two times to set rectangle"])
            elif but == BUT_BACKSPACE:
                curr_dict = keep_data(prev_dict)
            elif but == BUT_PLUS or but == BUT_MINUS:
                if prev_but != BUT_PLUS and prev_but != BUT_MINUS:
                    curr_dict = keep_data(prev_dict)
                    __prev_dict = keep_data(curr_dict)
                else:
                    curr_dict = keep_data(__prev_dict)
                curr_dict['class_val_to_dist'][curr_dict['class_val']] += 5 if but == BUT_PLUS else -5
                print('on_mouse_click(last_event={}, last_x={}, last_y={}, None, None)'.format(last_event, last_x, last_y))
                on_mouse_click(last_event, last_x, last_y, None, None)
                __prev_dict['class_val_to_dist'][curr_dict['class_val']] = curr_dict['class_val_to_dist'][curr_dict['class_val']]
            elif but == BUT_ESC:
                exit(0)
        elif but in classes_buttons:
            curr_dict['class_val'] = classes_buttons[but]
            if len(curr_dict['class_val_to_rect'][curr_dict['class_val']]) == 0:
                Z_WERE_PRESSED = True
                draw_status('CREATING RECTANGLE FOR "{}"'.format(class_val_to_str[curr_dict['class_val']]),
                            ["click in overlay window two times to set rectangle"])
            else:
                select_classes = copy.deepcopy(select_classes_hints)
                select_classes.extend(actions_with_classes_hints)
                draw_status('MARKUP "{}" CLASS PIXELS'.format(class_val_to_str[curr_dict['class_val']]), select_classes)
        prev_dict = keep_data(curr_dict)
        prev_but = but

    np.save(out_labeled_mask_path, curr_dict['labeled_mask'])
    get_overlay()
    cv2.imwrite(out_labeled_overlay_path, overlay)
    cv2.imwrite(out_image_path, origin_img)
