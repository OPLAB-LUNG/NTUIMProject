import pylidc
from pylidc.utils import consensus
import torch
import SimpleITK as sitk
import pandas
import glob, os
import numpy as np
import tqdm
import collections
from collections import namedtuple 

IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])

def masks_build(suid, hu_a):
    scans = {s.series_instance_uid:s for s in pylidc.query(pylidc.Scan).all()}
    s = scans[suid]
    ann_count = np.zeros_like(hu_a, dtype=int)
    for ann_cluster in s.cluster_annotations():
        # print(ann_cluster)
        for ann in ann_cluster:
            # print("id: ", ann.id)
            # print("shape: ", ann.boolean_mask().shape)
            mask = ann.boolean_mask()
            bbox = ann.bbox_matrix().T
            # print("bbox rci: ", bbox)
            
            bbox = np.roll(bbox, shift=1, axis=1)
            # print("bbox irc: ", bbox)
            result = np.diff(bbox, axis=0)[0]
            # print("ijk: ", result[0], result[1], result[2])
            result_p = np.diff(bbox, axis=0)[0] + 1
            # print("ijk: ", result_p[0], result_p[1], result_p[2])
            mask = np.transpose(mask, (2, 0, 1))
            # print("origin: ", ann_count[bbox[0][0]:bbox[0][0] + result_p[0], bbox[0][1]:bbox[0][1] + result_p[1], bbox[0][2]:bbox[0][2]+result_p[2]])
            ann_count[bbox[0][0]:bbox[0][0] + result_p[0], bbox[0][1]:bbox[0][1] + result_p[1], bbox[0][2]:bbox[0][2]+result_p[2]] += mask.astype(int)
            # print("add: ", ann_count[bbox[0][0]:bbox[0][0] + result_p[0], bbox[0][1]:bbox[0][1] + result_p[1], bbox[0][2]:bbox[0][2]+result_p[2]])
  
    masks = (ann_count >= 1)  
    
    return masks

def create_ann_maldf():
    annotations = pandas.read_csv('E:/LUNA/annotation/annotations.csv')
    malignancy_data = []
    missing = []
    spacing_dict = {}
    scans = {s.series_instance_uid:s for s in pylidc.query(pylidc.Scan).all()}
    suids = annotations.seriesuid.unique()
    
    cnt = 0
    
    for suid in tqdm.tqdm(suids):
        fn = glob.glob('E:/LUNA/Luna_Data/subset*/{}.mhd'.format(suid))
        if len(fn) == 0 or '*' in fn[0]:
            missing.append(suid)
            continue
        fn = fn[0]
        x = sitk.ReadImage(fn)
        spacing_dict[suid] = x.GetSpacing()
        s = scans[suid]
        for ann_cluster in s.cluster_annotations():
            cmask,cbbox,cmasks = consensus(ann_cluster, clevel=0.5)
            cbbox_details = calculate_box(cmask, cbbox)
            # print("cmask: ", cmask)
            # print("cbbox", cbbox, (cbbox[2]))
            # print("cmasks", len(cmasks))
            # this is our malignancy criteron described in Chapter 14
            is_malignant = len([a.malignancy for a in ann_cluster if a.malignancy >= 4])>=2
            centroid = np.mean([a.centroid for a in ann_cluster], 0)
            bbox = np.mean([a.bbox_matrix() for a in ann_cluster], 0).T
            coord = x.TransformIndexToPhysicalPoint([int(np.round(i)) for i in centroid[[1, 0, 2]]])
            bbox_low = x.TransformIndexToPhysicalPoint([int(np.round(i)) for i in bbox[0, [1, 0, 2]]])
            bbox_high = x.TransformIndexToPhysicalPoint([int(np.round(i)) for i in bbox[1, [1, 0, 2]]])
            #mask = ann.boolean_mask()
            malignancy_data.append((suid, coord[0], coord[1], coord[2], is_malignant, [a.malignancy for a in ann_cluster], [cbbox[2].start, cbbox[2].stop], cbbox_details))
        # break
        cnt += 1
        if cnt == 5:
            break
    print("MISSING", missing)
    df_mal = pandas.DataFrame(malignancy_data, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'mal_bool', 'mal_details', 'cbbox', 'cbbox_details'])
    return df_mal
    
def create_ann_malcsv(df_mal):
    annotations = pandas.read_csv('E:/LUNA/annotation/annotations.csv')
    processed_annot = []
    annotations['mal_bool'] = float('nan')
    annotations['mal_details'] = [[] for _ in annotations.iterrows()]
    # bbox_keys = ['bboxLowX', 'bboxLowY', 'bboxLowZ', 'bboxHighX', 'bboxHighY', 'bboxHighZ']
    # for k in bbox_keys:
    #     annotations[k] = float('nan')
    annotations['cbbox'] =  [[] for _ in annotations.iterrows()]
    annotations['cbbox_details'] = [[[]] for _ in annotations.iterrows()]
    for series_id in tqdm.tqdm(annotations.seriesuid.unique()):
        # series_id = '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860'
        # c = candidates[candidates.seriesuid == series_id]
        a = annotations[annotations.seriesuid == series_id]
        m = df_mal[df_mal.seriesuid == series_id]
        if len(m) > 0: 
            m_ctrs = m[['coordX', 'coordY', 'coordZ']].values
            a_ctrs = a[['coordX', 'coordY', 'coordZ']].values
            #print(m_ctrs.shape, a_ctrs.shape)
            matches = (np.linalg.norm(a_ctrs[:, None] - m_ctrs[None], ord=2, axis=-1) / a.diameter_mm.values[:, None] < 0.5)
            has_match = matches.max(-1)
            match_idx = matches.argmax(-1)[has_match]
            a_matched = a[has_match].copy()
            # c_matched['diameter_mm'] = a.diameter_mm.values[match_idx]
            a_matched['mal_bool'] = m.mal_bool.values[match_idx]
            a_matched['mal_details'] = m.mal_details.values[match_idx]
            a_matched['cbbox'] = m.cbbox.values[match_idx]
            a_matched['cbbox_details'] = m.cbbox_details.values[match_idx]
            #a_matched['idx'] = m.id.values[match_idx]
            #for k in bbox_keys:
                #a_matched[k] = m[k].values[match_idx]
            processed_annot.append(a_matched)
            processed_annot.append(a[~has_match])
        
        # else:
        #     processed_annot.append(c)
        
    processed_annot = pandas.concat(processed_annot)
    processed_annot.sort_values('mal_bool', ascending=False, inplace=True)
    processed_annot['len_mal_details'] = processed_annot.mal_details.apply(len)
    df_nona = processed_annot.dropna()
    df_nona.to_csv('./annotations_object_detection_1127_5.csv', index=False)


def calculate_box(cmask, cbbox):
    #print(cbbox)
    
    cmask_reshape = np.transpose(np.array(cmask), (2, 1, 0))
    #print(cmask_reshape)
    ret = []
    
    for z_mask in cmask_reshape:
        min_x, max_x, min_y, max_y = 511, 0, 511, 0
        for i in range(len(z_mask)):
            for j in range(len(z_mask[i])):
                if z_mask[i][j]:
                    if i <= min_y:
                        min_y = i
                    if j <= min_x:
                        min_x = j
                    if i >= max_y:
                        max_y = i
                    if j >= max_x:
                        max_x = j
        #print("cbbox:", cbbox)
        #print("minx, maxx, miny, maxy: ", (min_x, max_x, min_y, max_y))
        if min_x == 511 and max_x == 0 and min_y == 511 and max_y == 0:
            ret.append('nan')
        else:
            center_x = round(((min_x + max_x) / 2 + cbbox[0].start) / 511, 10)
            center_y = round(((min_y + max_y) / 2 + cbbox[1].start) / 511, 10)
            width_x = round((max_x - min_x + 4) / 512, 10)
            height_y = round((max_y - min_y + 4) / 512, 10)
            #print("centerx, centery, width, height: ", center_x, center_y, width_x, height_y)
            ret.append([center_y, center_x, height_y, width_x])
        
        #print(ret)
        
    return ret
                    
    
    

#create_ann_malcsv(create_ann_maldf())
