import numpy as np

# PASCAL VOC评价Detection结果的代码
def voc_ap(self, rec, prec, use_07_metric=True):
    if use_07_metric:
        ap = 0.
        # 2010年以前按recall等间隔取11个不同点处的精度值做平均(0., 0.1, 0.2, …, 0.9, 1.0)
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                # 取最大值等价于2010以后先计算包络线的操作，保证precise非减
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # 2010年以后取所有不同的recall对应的点处的精度值做平均
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # 计算包络线，从后往前取最大保证precise非减
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # 找出所有检测结果中recall不同的点
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        # 用recall的间隔对精度作加权平均
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# 计算每个类别对应的AP，mAP是所有类别AP的平均值
def voc_eval(self, detpath,
             classname,
             ovthresh=0.5,
             use_07_metric=True):
    # 提取所有测试图片中当前类别所对应的所有ground_truth
    class_recs = {}
    npos = 0
    # 遍历所有测试图片
    for imagename in imagenames:
        # 找出所有当前类别对应的object
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        # 该图片中该类别对应的所有bbox
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        # 该图片中该类别对应的所有bbox的是否已被匹配的标志位
        det = [False] * len(R)
        # 累计所有图片中的该类别目标的总数，不算diffcult
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # 读取相应类别的检测结果文件，每一行对应一个检测目标
    if any(lines) == 1:
        # 某一行对应的检测目标所属的图像名
        image_ids = [x[0] for x in splitlines]
        # 读取该目标对应的置信度
        confidence = np.array([float(x[1]) for x in splitlines])
        # 读取该目标对应的bbox
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # 将该类别的检测结果按照置信度大小降序排列
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        # 该类别检测结果的总数（所有检测出的bbox的数目）
        nd = len(image_ids)
        # 用于标记每个检测结果是tp还是fp
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        # 按置信度遍历每个检测结果
        for d in range(nd):
            # 取出该条检测结果所属图片中的所有ground truth
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            # 计算与该图片中所有ground truth的最大重叠度
            if BBGT.size > 0:
                ......
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            # 如果最大的重叠度大于一定的阈值
            if ovmax > ovthresh:
                # 如果最大重叠度对应的ground truth为difficult就忽略
                if not R['difficult'][jmax]:
                    # 如果对应的最大重叠度的ground truth以前没被匹配过则匹配成功，即tp
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    # 若之前有置信度更高的检测结果匹配过这个ground truth，则此次检测结果为fp
                    else:
                        fp[d] = 1.
            # 该图片中没有对应类别的目标ground truth或者与所有ground truth重叠度都小于阈值
            else:
                fp[d] = 1.

        # 按置信度取不同数量检测结果时的累计fp和tp
        # np.cumsum([1, 2, 3, 4]) -> [1, 3, 6, 10]
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # 召回率为占所有真实目标数量的比例，非减的，注意npos本身就排除了difficult，因此npos=tp+fn
        rec = tp / float(npos)
        # 精度为取的所有检测结果中tp的比例
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # 计算recall-precise曲线下面积（严格来说并不是面积）
        ap = self.voc_ap(rec, prec, use_07_metric)
    # 如果这个类别对应的检测结果为空，那么都是-1
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap
