import pylidc
import torch
import SimpleITK as sitk
import pandas
import glob, os
import numpy as np
import tqdm
import pylidc
import collections
from collections import namedtuple

IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])

def masks_build(suid, hu_a):
    scans = {s.series_instance_uid:s for s in pylidc.query(pylidc.Scan).all()}
    s = scans[suid]
    ann_count = np.zeros_like(hu_a, dtype=int)
    for ann_cluster in s.cluster_annotations():
        # print(ann_cluster)
        if len(ann_cluster) < 3:
            continue
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

def ann_group(suid, hu_a):
    scans = {s.series_instance_uid:s for s in pylidc.query(pylidc.Scan).all()}
    s = scans[suid]
    ann_data= []
    for i, ann_cluster in enumerate(s.cluster_annotations()):
        upper_ndx = []
        lower_ndx = []
        for ann in ann_cluster:

            bbox = ann.bbox_matrix().T
            bbox = np.roll(bbox, shift=1, axis=1)
            upper_ndx.append(bbox[0][0])
            lower_ndx.append(bbox[1][0])
        # print("upper: ", upper_ndx)
        # print("lower: ", lower_ndx)
        ann_data.append((suid, i, min(upper_ndx), max(lower_ndx)))
    return ann_data

def ann_counter(suid, hu_a):
    scans = {s.series_instance_uid:s for s in pylidc.query(pylidc.Scan).all()}
    s = scans[suid]
    counter = 0
    for i, ann_cluster in enumerate(s.cluster_annotations()):
        if len(ann_cluster) >= 2:
            counter += 1
    return counter
    

def create_ann_maldf():
    annotations = pandas.read_csv('C:/Users/oplab/Desktop/Luna16_data/annotations.csv')
    malignancy_data = []
    missing = []
    spacing_dict = {}
    scans = {s.series_instance_uid:s for s in pylidc.query(pylidc.Scan).all()}
    suids = annotations.seriesuid.unique()
    for suid in tqdm.tqdm(suids):
        fn = glob.glob('C:/Users/oplab/Desktop/Luna16_data/Luna16_img/subset*/{}.mhd'.format(suid))
        if len(fn) == 0 or '*' in fn[0]:
            missing.append(suid)
            continue
        fn = fn[0]
        x = sitk.ReadImage(fn)
        spacing_dict[suid] = x.GetSpacing()
        s = scans[suid]
        for ann_cluster in s.cluster_annotations():
            # this is our malignancy criteron described in Chapter 14
            is_malignant = len([a.malignancy for a in ann_cluster if a.malignancy >= 4])>=2
            centroid = np.mean([a.centroid for a in ann_cluster], 0)
            bbox = np.mean([a.bbox_matrix() for a in ann_cluster], 0).T
            coord = x.TransformIndexToPhysicalPoint([int(np.round(i)) for i in centroid[[1, 0, 2]]])
            bbox_low = x.TransformIndexToPhysicalPoint([int(np.round(i)) for i in bbox[0, [1, 0, 2]]])
            bbox_high = x.TransformIndexToPhysicalPoint([int(np.round(i)) for i in bbox[1, [1, 0, 2]]])
            mask = ann.boolean_mask()
            malignancy_data.append((idx, suid, coord[0], coord[1], coord[2], bbox_low[0], bbox_low[1], bbox_low[2], bbox_high[0], bbox_high[1], bbox_high[2], is_malignant, [a.malignancy for a in ann_cluster]))
    print("MISSING", missing)
    df_mal = pandas.DataFrame(malignancy_data, columns=['id', 'seriesuid', 'coordX', 'coordY', 'coordZ', 'bboxLowX', 'bboxLowY', 'bboxLowZ', 'bboxHighX', 'bboxHighY', 'bboxHighZ', 'mal_bool', 'mal_details'])
    return df_mal
    
def create_ann_malcsv(df_mal):
    annotations = pandas.read_csv('C:/Users/oplab/Desktop/Luna16_data/annotations.csv')
    processed_annot = []
    annotations['mal_bool'] = float('nan')
    annotations['mal_details'] = [[] for _ in annotations.iterrows()]
    bbox_keys = ['bboxLowX', 'bboxLowY', 'bboxLowZ', 'bboxHighX', 'bboxHighY', 'bboxHighZ']
    for k in bbox_keys:
        annotations[k] = float('nan')
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
            a_matched['idx'] = m.id.values[match_idx]
            for k in bbox_keys:
                a_matched[k] = m[k].values[match_idx]
            processed_annot.append(a_matched)
            processed_annot.append(a[~has_match])
        else:
            processed_annot.append(c)
    processed_annot = pandas.concat(processed_annot)
    processed_annot.sort_values('mal_bool', ascending=False, inplace=True)
    processed_annot['len_mal_details'] = processed_annot.mal_details.apply(len)
    df_nona = processed_annot.dropna()
    df_nona.to_csv('./annotations_with_malignancy_new.csv', index=False)