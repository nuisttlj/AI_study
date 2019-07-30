# Dataset and Network cfg
IMG_WIDTH = 416
IMG_HEIGHT = 416

CLASS_NUM = 80

ANCHORS_GROUP = {
    13: [[116, 90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]]
}

# Net cfg
cfg_fms52 = ['C332', 'D', 'R1', 'D', 'R2', 'D', 'R8']
in_channels_fms52 = 3

cfg_fms26 = ['D', 'R8']
in_channels_fms26 = 256

cfg_p1 = ['D', 'R4', 'C1512', 'C31024', 'C1512', 'C31024', 'C1512', 'C31024', 'O255']
in_channels_p1 = 512

cfg_up1 = ['D', 'R4', 'C1512', 'C31024', 'C1512', 'C31024', 'C1512', 'C1256', 'U']
in_channels_up1 = 512

cfg_p2 = ['C1256', 'C3768', 'C1256', 'C3768', 'C1256', 'C3512', 'O255']
in_channels_p2 = 768

cfg_up2 = ['C1256', 'C3768', 'C1256', 'C3768', 'C1256', 'C1128', 'U']
in_channels_up2 = 768

cfg_p3 = ['C1128', 'C3384', 'C1128', 'C3384', 'C1128', 'C3256', 'O255']
in_channels_p3 = 384
