import collections
from collections import namedtuple  
import glob
import SimpleITK as sitk
from torch.utils.data import Dataset
import os
import csv
import numpy as np
from util import XyzTuple, xyz2irc, logging, getCache
# from util import XyzTuple, xyz2irc, logging
import functools
import random
import torch
import functools
from segmentation import segment, init_segment_model
from augmentation import augment, init_augment_model
from pylidc_func import masks_build

# raw_cache = getCache('nodule_segmentation_fullCT')
# raw_cache = getCache('nodule_segmentation_fullCT')
# raw_cache = getCache('augmented_segmented')
# raw_cache = getCache("augemented_segmented")
# raw_cache = getCache("augmented_reduce")
# raw_cache = getCache("origin_contour_5")
# raw_cache = getCache("origin_contour")
# raw_cache = getCache("seg_contour")
# raw_cache = getCache("aug_contour")
raw_cache = getCache("augseg_contour")
# raw_cache = getCache("augmented_reduce_box")
# raw_cache = getCache("segmented_reduce")
# raw_cache = getCache("augmented_subset0")
# raw_cache = getCache("segmented_subset0")


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# getCandidateInfoList
CandidateInfoTuple = namedtuple('CandidateInfoTuple', 'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')
# raw_cache = getCache('part2ch13_raw_test')
@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    # mhd_list = glob.glob('D:/LUNA/Luna16_AugData/subset*/*.mhd')
    mhd_list = glob.glob('D:/LUNA/Luna16_AugSegData/subset*/*.mhd')
    # mhd_list = glob.glob('D:/LUNA/Luna16_SegData/subset*/*.mhd')
    # mhd_list = glob.glob('C:/Users/oplab/Desktop/Luna16_data/Luna16_img/subset*/*.mhd')
    # mhd_list = glob.glob('D:/LUNA/Luna16_AugData/subset0/*.mhd')
    # print(mhd_list)
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidateInfo_list = []
    with open('C:/Users/oplab/Desktop/Luna16_data/annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            isMal_bool = {'False': False, 'True': True}[row[5]] #it record the malignancy or not

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            candidateInfo_list.append(
                CandidateInfoTuple(
                    True,
                    True,
                    isMal_bool,
                    annotationDiameter_mm,
                    series_uid,
                    annotationCenter_xyz,
                )
            )
            
    # print("candidateInfo_list: ", len(candidateInfo_list))

    # print("candidateInfo_list: ", len(candidateInfo_list))
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

# getCandidateInfoDict
@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):  #把candidateInfoList包成Dict
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid,
                                      []).append(candidateInfo_tup)

    return candidateInfo_dict

# getCt
@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

# getCtSampleSize
@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):  #回傳該uid的slice size
    ct = Ct(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes

@raw_cache.memoize(typed=True)
def getFullCT(series_uid, center_xyz, contextSlices_count):
    ct = getCt(series_uid)
    ct_t, pos_t, slice_ndx = ct.getRawFullCT(center_xyz, contextSlices_count)
    return ct_t, pos_t, slice_ndx

@raw_cache.memoize(typed=True)
def getCTSlice(series_uid, slice_ndx, contextSlices_count):
    ct = getCt(series_uid)
    ct_t, pos_t = ct.getRawCTSlice(slice_ndx, contextSlices_count)
    return ct_t, pos_t

# getCtRawCandidate
@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz,
                                                         width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, pos_chunk, center_irc

class Ct:
    def __init__(self, series_uid):
        # mhd_path = glob.glob(
        #     # '../data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid))[0]
        #     '../Luna_Data/subset*/{}.mhd'.format(series_uid))[0]
        mhd_path = glob.glob('D:/LUNA/Luna16_AugSegData/subset*/{}.mhd'.format(series_uid))
        # mhd_path = glob.glob('D:/LUNA/Luna16_SegData/subset*/{}.mhd'.format(series_uid))
        mask_path = glob.glob('C:/LUNA/Luna16_AugMask/subset*/{}.mhd'.format(series_uid))
        # mhd_path = glob.glob('C:/Users/oplab/Desktop/Luna16_data/Luna16_img/subset*/{}.mhd'.format(series_uid))
        # mhd_path = glob.glob('D:/LUNA/Luna16_AugData/subset0/{}.mhd'.format(series_uid))
        # mhd_path = glob.glob('D:/LUNA/Luna16_SegData/subset*/{}.mhd'.format(series_uid))
        
        # print(mhd_path)

        ct_mhd = sitk.ReadImage(mhd_path)
        if ct_mhd.GetDimension()==4 and ct_mhd.GetSize()[3]==1:
            ct_mhd = ct_mhd[...,0]
        origin = sitk.GetArrayFromImage(ct_mhd)
        mask_mhd = sitk.ReadImage(mask_path)
        if mask_mhd.GetDimension()==4 and mask_mhd.GetSize()[3]==1:
            mask_mhd = mask_mhd[...,0]
        mask = sitk.GetArrayFromImage(mask_mhd)
        # model_best, device = init_augment_model()
        # augmented = augment(origin, model_best, device)
        # model, device = init_segment_model()
        # segmented = segment(augmented, model, device)
        
        # print(mhd_path)
        # augmented = ct_augment(origin)
        # segmented = ct_segment(augmented)
        # self.hu_a = np.array(segmented, dtype=np.float32)
        # self.hu_a = np.array(augmented, dtype=np.float32)
        self.hu_a = np.array(origin, dtype=np.float32)
        self.positive_mask = np.array(mask, dtype=np.bool)
        # self.hu_a = np.array(segmented, dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.

        self.series_uid = series_uid
        # if ("1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492" == series_uid):
        #     print(*ct_mhd.GetOrigin())
        #     print(*ct_mhd.GetSpacing())
        #     print(*ct_mhd.GetDirection())

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        # x, y, z = ct_mhd.GetSpacing()
        # self.vxSize_xyz = XyzTuple(x, y, z/2)
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        candidateInfo_list = getCandidateInfoDict()[self.series_uid]

        self.positiveInfo_list = [
            candidate_tup
            for candidate_tup in candidateInfo_list
            if candidate_tup.isNodule_bool
        ] #只將nodule放入list中
        # self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list) #mask的大小和hu_a一致
        # self.positive_mask = masks_build(self.series_uid, self.hu_a)
        self.positive_indexes = (self.positive_mask.sum(axis=(1,2)) #axis=(1,2)是將row和column上的所有true加起來
                                 .nonzero()[0].tolist())  #會將所有有被標記的記進來
        # print(self.positive_indexes)

    #為了幫所有的pixel記上是否為nodule的label，我們需要mask，並使用threshold的方式來框出哪裡是nodule而哪裡不是
    def buildAnnotationMask(self, positiveInfo_list, threshold_hu = -700):
        boundingBox_a = np.zeros_like(self.hu_a, dtype=bool) # all False tensor

        for candidateInfo_tup in positiveInfo_list: #traverse所有的nodules
            center_irc = xyz2irc(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )
            ci = int(center_irc.index)  # the center of voxel
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                # 從index找，看哪裡會遇到空氣，當兩邊都遇到空氣後就把邊界設在大的那段
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                # 從row找，看哪裡會遇到空氣，當兩邊都遇到空氣後就把邊界設在大的那段
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                # 從column找，看哪裡會遇到空氣，當兩邊都遇到空氣後就把邊界設在大的那段
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            # assert index_radius > 0, repr([candidateInfo_tup.center_xyz, center_irc, self.hu_a[ci, cr, cc]])
            # assert row_radius > 0
            # assert col_radius > 0

            boundingBox_a[
                 ci - index_radius: ci + index_radius + 1,
                 cr - row_radius: cr + row_radius + 1,
                 cc - col_radius: cc + col_radius + 1] = True #將box裡的所有格子設成TRUE

        mask_a = boundingBox_a & (self.hu_a > threshold_hu)  #最後會對box和threshold低於-700的值做and
        # mask_a = boundingBox_a

        return mask_a
    def getRawCTSlice(self, slice_ndx, contextSlices_count):
        ct_t = torch.zeros((contextSlices_count * 2 + 1, 512, 512))  #預設是上下兩張

        start_ndx = slice_ndx - contextSlices_count
        end_ndx = slice_ndx + contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0) #避免邊界，遇到邊界會重複
            context_ndx = min(context_ndx, self.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(self.hu_a[context_ndx].astype(np.float32))

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(self.positive_mask[slice_ndx]).unsqueeze(0)
        return ct_t, pos_t
    
    def getRawFullCT(self, center_xyz, contextSlices_count):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz,
                             self.direction_a)
        slice_ndx = center_irc.index
        ct_a = self.hu_a
        pos_a = self.positive_mask
        
        ct_t = torch.zeros((contextSlices_count * 2 + 1, 512, 512))  #預設是上下兩張

        start_ndx = slice_ndx - contextSlices_count
        end_ndx = slice_ndx + contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0) #避免邊界，遇到邊界會重複
            context_ndx = min(context_ndx, ct_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct_a[context_ndx].astype(np.float32))

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(pos_a[slice_ndx]).unsqueeze(0)
        
        # print(self.series_uid)
        # print(ct_t.size())
        # print(pos_t.size())
        
        return ct_t, pos_t, slice_ndx

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz,
                             self.direction_a)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc  #pos_chunk 就是mask
    
class Luna2dSegmentationDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 set_class=None,
                 series_uid=None,
                 contextSlices_count=3,
                 fullCt_bool=False,
            ):
        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(getCandidateInfoDict().keys())

        if set_class == "Validation":
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride] #將uid存進series_list，每val_stride個記一次次
            assert self.series_list
        elif set_class == "Testing":
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[1::val_stride]
        elif set_class == "Training":
            slist = self.series_list
            # 將列表分為每10個一組的子列表
            sublists = [slist[i:i+10] for i in range(0, len(slist), 10)]

            # 移除每個子列表的第1和第2個元素
            modified_sublists = [sublist[2:] for sublist in sublists]

            # 將修改後的子列表組合成新的列表
            self.series_list = [item for sublist in modified_sublists for item in sublist]
            assert self.series_list
        elif set_class == "All":
            self.series_list = self.series_list

        self.sample_list = [] #裝所有會進sample的uid及該slice index

        for series_uid in self.series_list:
            index_count, positive_indexes = getCtSampleSize(series_uid) #index count 是總index數
            #positive indexes 是有nodule mask的index

            if self.fullCt_bool:  #fullCt_bool 為true代表將整個nodule放入
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in range(index_count)]
            else: #fullCt_bool 為false代表只將positive mask放入
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in positive_indexes]

        self.candidateInfo_list = getCandidateInfoList()

        series_set = set(self.series_list)  #將series_list做成set
        self.candidateInfo_list = [cit for cit in self.candidateInfo_list
                                   if cit.series_uid in series_set] #將在seires_set的candidateInfor_list放入

        self.pos_list = [nt for nt in self.candidateInfo_list
                            if nt.isNodule_bool]  #只裝是nodule的部分

        log.info("{!r}: {} {} series, {} candidates, {} slices, {} nodules".format(
            self,
            len(self.series_list),
            {None: "general", "Testing": 'testing', "Validation": 'validation', "Training": 'training', "All": "all"}[set_class],
            len(self.candidateInfo_list),
            len(self.sample_list),
            len(self.pos_list),
        ))

        # 540 training series, 15774 slices, 1054 nodules
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx): #for validation
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)] #epoch size, sample list 會放uid和slice ndx
        # series_uid, slice_ndx = self.sample_list[ndx]
        return self.getitem_fullSlice(series_uid, slice_ndx)
    
    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct_t, pos_t = getCTSlice(series_uid, slice_ndx, self.contextSlices_count)

        return ct_t, pos_t, series_uid, slice_ndx

#     def getitem_fullSlice(self, series_uid, slice_ndx):
#         ct = getCt(series_uid)
#         ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))  #預設是上下兩張

#         start_ndx = slice_ndx - self.contextSlices_count
#         end_ndx = slice_ndx + self.contextSlices_count + 1
#         for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
#             context_ndx = max(context_ndx, 0) #避免邊界，遇到邊界會重複
#             context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
#             ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

#         # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
#         # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
#         # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
#         # The upper bound nukes any weird hotspots and clamps bone down
#         ct_t.clamp_(-1000, 1000)

#         pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

#         return ct_t, pos_t, ct.series_uid, slice_ndx
    
class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset): #training dataset會將原CT裁切成64*64，而這64*64來自center附近的96*96
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2

    def __len__(self):
        # return len(self.sample_list) * 5
        # return 300000
        # return 30000
        return len(self.sample_list)
        # return len(self.pos_list)
        # return 4

    def shuffleSamples(self):
        random.shuffle(self.candidateInfo_list)
        random.shuffle(self.pos_list)
        random.shuffle(self.sample_list)

    def __getitem__(self, ndx): #我們只從isNodule的部分取樣本
        # candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)]
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
        # return self.getitem_trainingCrop(candidateInfo_tup)
        return self.getitem_fullSlice(series_uid, slice_ndx)
        # return self.getitem_trainingCrop()

    # def getitem_trainingCrop(self, candidateInfo_tup):
    #     ct_t, pos_t, slice_ndx = getFullCT(candidateInfo_tup.series_uid, candidateInfo_tup.center_xyz, self.contextSlices_count)
    #     series_uid = candidateInfo_tup.series_uid
    #     return ct_t, pos_t, series_uid, slice_ndx
    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct_t, pos_t = getCTSlice(series_uid, slice_ndx, self.contextSlices_count)

        return ct_t, pos_t, series_uid, slice_ndx
    
class PrepcacheLunaDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.candidateInfo_list = getCandidateInfoList()
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]
        
        self.series_list = sorted(getCandidateInfoDict().keys())
        
        self.sample_list = [] #裝所有會進sample的uid及該slice index
        
        self.fullCt_bool = False

        for series_uid in self.series_list:
            index_count, positive_indexes = getCtSampleSize(series_uid) #index count 是總index數
            #positive indexes 是有nodule mask的index

            if self.fullCt_bool:  #fullCt_bool 為true代表將整個nodule放入
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in range(index_count)]
            else: #fullCt_bool 為false代表只將positive mask放入
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in positive_indexes]

        self.seen_set = set()
        self.candidateInfo_list.sort(key=lambda x: x.series_uid)

    def __len__(self):
        return len(self.sample_list)

#     def __getitem__(self, ndx):
#         # candidate_t, pos_t, series_uid, center_t = super().__getitem__(ndx)
#         candidateInfo_tup = self.candidateInfo_list[ndx]
#         series_uid = candidateInfo_tup.series_uid
#         if series_uid not in self.seen_set:
#             self.seen_set.add(series_uid)
#             getCtSampleSize(series_uid)
# #             ct = getCt(series_uid)
# #             for mask_ndx in ct.positive_indexes:
# #                 build2dLungMask(series_uid, mask_ndx)

#         # getFullCT(candidateInfo_tup.series_uid, candidateInfo_tup.center_xyz, 1)

        return 0, 1 #candidate_t, pos_t, series_uid, center_t
    def __getitem__(self, ndx):
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
        
        getCTSlice(series_uid, slice_ndx, 3)
        
        # candidate_t, pos_t, series_uid, center_t = super().__getitem__(ndx)

#         candidateInfo_tup = self.candidateInfo_list[ndx]
#         # getFullCT(candidateInfo_tup.series_uid, candidateInfo_tup.center_xyz, 3)

#         series_uid = candidateInfo_tup.series_uid
        if series_uid not in self.seen_set:
            self.seen_set.add(series_uid)

            getCtSampleSize(series_uid)
#             ct = getCt(series_uid)
            # for mask_ndx in ct.positive_indexes:
            #     build2dLungMask(series_uid, mask_ndx)

        return 0, 1 #candidate_t, pos_t, series_uid, center_t