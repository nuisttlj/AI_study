import torch.nn.functional as F
import torch

dropout_rate = 0.5


def standard_unit(input_tensor, inchannel, outchannel, kernel_size=3, padding=1):
    w1 = torch.empty(outchannel, inchannel, kernel_size, kernel_size)
    b1 = torch.empty(outchannel)
    torch.nn.init.kaiming_normal_(w1)
    torch.nn.init.kaiming_normal_(b1)
    x = F.conv2d(input_tensor, w1, b1, padding=padding)
    x = F.batch_norm(x)
    x = F.prelu(x)
    x = F.dropout2d(x, dropout_rate)
    w2 = torch.empty(outchannel, outchannel, kernel_size, kernel_size)
    b2 = torch.empty(outchannel)
    torch.nn.init.kaiming_normal_(w2)
    torch.nn.init.kaiming_normal_(b2)
    x = F.conv2d(x, w1, b1, padding=padding)
    x = F.batch_norm(x)
    x = F.prelu(x)
    x = F.dropout2d(x, dropout_rate)

    return x


########################################

"""
Standard UNet++ [Zhou et.al, 2018]
Total params: 9,041,601
"""


def Nest_Net(img_input, num_class=1, deep_supervision=False):
    nb_filter = [32, 64, 128, 256, 512]

    conv1_1 = standard_unit(img_input, nb_filter=nb_filter[0])
    pool1 = F.max_pool2d(conv1_1, (2, 2), strides=(2, 2))

    conv2_1 = standard_unit(pool1, nb_filter=nb_filter[1])
    pool2 = F.max_pool2d(conv2_1, (2, 2), strides=(2, 2))

    # 可以尝试线性插值接一个1*1卷积压缩通道
    # up1_2 = F.interpolate(conv2_1, 2, mode='bilinear')

    up1_2 = F.conv_transpose2d(conv2_1, (nb_filter[0], (2, 2), strides=(2, 2), padding='same')
    conv1_2 = torch.cat([up1_2, conv1_1], dim=1)
    conv1_2 = standard_unit(conv1_2, nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, nb_filter=nb_filter[2])
    pool3 = F.max_pool2d(conv3_1, (2, 2), strides=(2, 2), name='pool3')

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = torch.cat([up2_2, conv2_1], axis=1)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=1)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = F.max_pool2d(conv4_1, (2, 2), strides=(2, 2), name='pool4')

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=1)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=1)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=1)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=1)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=1)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=1)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(input=img_input, output=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(input=img_input, output=[nestnet_output_4])

    return model
