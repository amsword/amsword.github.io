---
layout: post
comments: true
title: Coco Evaluation for the Bounding Box
published: false
---

# Abstract

The code is based on pycocotools and the maskrcnn-benchmark.

# Details
## prepare the prediction results.
   ```python
   coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset)
   def prepare_for_coco_detection(predictions, dataset):
       # assert isinstance(dataset, COCODataset)
       coco_results = []
       for image_id, prediction in enumerate(predictions):
           original_id = dataset.id_to_img_map[image_id]
           if len(prediction) == 0:
               continue
   
           # TODO replace with get_img_info?
           image_width = dataset.coco.imgs[original_id]["width"]
           image_height = dataset.coco.imgs[original_id]["height"]
           prediction = prediction.resize((image_width, image_height))
           prediction = prediction.convert("xywh")
   
           boxes = prediction.bbox.tolist()
           scores = prediction.get_field("scores").tolist()
           labels = prediction.get_field("labels").tolist()
   
           mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
   
           coco_results.extend(
               [
                   {
                       "image_id": original_id,
                       "category_id": mapped_labels[k],
                       "bbox": box,
                       "score": scores[k],
                   }
                   for k, box in enumerate(boxes)
               ]
           )
       return coco_results
   ``` 
## save the coco_results to the json file and load it by Coco in pycocotools lib
   ```python
   with open(json_result_file, "w") as f:
       json.dump(coco_results, f)
   coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()
   class COCO:
       def loadRes(self, resFile):
           """
           removed some non-related code path
           Load result file and return a result api object.
           :param   resFile (str)     : file name of result file
           :return: res (obj)         : result api object
           """
           res = COCO()
           res.dataset['images'] = [img for img in self.dataset['images']]
           if type(resFile) == str or type(resFile) == unicode:
               anns = json.load(open(resFile))
           annsImgIds = [ann['image_id'] for ann in anns]
           if 'bbox' in anns[0] and not anns[0]['bbox'] == []:
               res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
               for id, ann in enumerate(anns):
                   bb = ann['bbox']
                   x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                   if not 'segmentation' in ann:
                       ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                   ann['area'] = bb[2]*bb[3]
                   ann['id'] = id+1
                   ann['iscrowd'] = 0
           res.dataset['annotations'] = anns
           res.createIndex()
           return res
   ```

## construct the `COCOEval` object
   ```python
   coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
   ```

## evaluate. 
   Calculate the IoU matrix between any ground-truth and the top-100 detections for each (image, cat). Two notes
    - If a ground-truth box is labeled as crowded, the IoU is
    Intersection/AreaOfDetection not Intersection/Union.
    - If a ground-truth box is labeled as crowded, this ground-truth box can be
    used more than once to match the predictions.

   ```python
   coco_eval.evaluate()
   class COCOeval:
       def evaluate(self):
           '''
           Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
           :return: None
           '''
           p = self.params
           p.imgIds = list(np.unique(p.imgIds))
           if p.useCats:
               p.catIds = list(np.unique(p.catIds))
           p.maxDets = sorted(p.maxDets)
           self.params=p
   
           self._prepare()
           # loop through images, area range, max detection number
           catIds = p.catIds if p.useCats else [-1]
   
           if p.iouType == 'segm' or p.iouType == 'bbox':
               computeIoU = self.computeIoU
           self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                           for imgId in p.imgIds
                           for catId in catIds}
   
           evaluateImg = self.evaluateImg
           maxDet = p.maxDets[-1]
           self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                    for catId in catIds
                    for areaRng in p.areaRng
                    for imgId in p.imgIds
                ]
           self._paramsEval = copy.deepcopy(self.params)
           toc = time.time()
           print('DONE (t={:0.2f}s).'.format(toc-tic))
   
       def _prepare(self):
           '''
           Prepare ._gts and ._dts for evaluation based on params
           :return: None
           '''
           def _toMask(anns, coco):
               # modify ann['segmentation'] by reference
               for ann in anns:
                   rle = coco.annToRLE(ann)
                   ann['segmentation'] = rle
           p = self.params
           if p.useCats:
               gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
               dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
           else:
               gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
               dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))
   
           # convert ground truth to mask if iouType == 'segm'
           if p.iouType == 'segm':
               _toMask(gts, self.cocoGt)
               _toMask(dts, self.cocoDt)
           # set ignore flag
           for gt in gts:
               gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
               gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
               if p.iouType == 'keypoints':
                   gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
           self._gts = defaultdict(list)       # gt for evaluation
           self._dts = defaultdict(list)       # dt for evaluation
           for gt in gts:
               self._gts[gt['image_id'], gt['category_id']].append(gt)
           for dt in dts:
               self._dts[dt['image_id'], dt['category_id']].append(dt)
           self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
           self.eval     = {}                  # accumulated evaluation results
   
       def computeIoU(self, imgId, catId):
           p = self.params
           if p.useCats:
               gt = self._gts[imgId,catId]
               dt = self._dts[imgId,catId]
           if len(gt) == 0 and len(dt) ==0:
               return []
           inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
           dt = [dt[i] for i in inds]
           if len(dt) > p.maxDets[-1]:
               dt=dt[0:p.maxDets[-1]]
   
           g = [g['bbox'] for g in gt]
           d = [d['bbox'] for d in dt]
   
           # compute iou between each dt and gt region
           iscrowd = [int(o['iscrowd']) for o in gt]
           ious = maskUtils.iou(d,g,iscrowd)
           return ious
   
       def evaluateImg(self, imgId, catId, aRng, maxDet):
           '''
           perform evaluation for single category and image
           :return: dict (single image results)
           '''
           p = self.params
           if p.useCats:
               gt = self._gts[imgId,catId]
               dt = self._dts[imgId,catId]
           if len(gt) == 0 and len(dt) ==0:
               return None
   
           for g in gt:
               if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                   g['_ignore'] = 1
               else:
                   g['_ignore'] = 0
   
           gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
           gt = [gt[i] for i in gtind]
           dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
           dt = [dt[i] for i in dtind[0:maxDet]]
           iscrowd = [int(o['iscrowd']) for o in gt]
           ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
   
           T = len(p.iouThrs)
           G = len(gt)
           D = len(dt)
           gtm  = np.zeros((T,G))
           dtm  = np.zeros((T,D))
           gtIg = np.array([g['_ignore'] for g in gt])
           dtIg = np.zeros((T,D))
           if not len(ious)==0:
               for tind, t in enumerate(p.iouThrs):
                   for dind, d in enumerate(dt):
                       # information about best match so far (m=-1 -> unmatched)
                       iou = min([t,1-1e-10])
                       m   = -1
                       for gind, g in enumerate(gt):
                           # if this gt already matched, and not a crowd, continue
                           if gtm[tind,gind]>0 and not iscrowd[gind]:
                               continue
                           # if dt matched to reg gt, and on ignore gt, stop
                           if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                               break
                           # continue to next gt unless better match made
                           if ious[dind,gind] < iou:
                               continue
                           # if match successful and best so far, store appropriately
                           iou=ious[dind,gind]
                           m=gind
                       # if match made store id of match for both dt and gt
                       if m ==-1:
                           continue
                       dtIg[tind,dind] = gtIg[m]
                       dtm[tind,dind]  = gt[m]['id']
                       gtm[tind,m]     = d['id']
           # set unmatched detections outside of area range to ignore
           a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
           dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
           # store results for given image and category
           return {
                   'image_id':     imgId,
                   'category_id':  catId,
                   'aRng':         aRng,
                   'maxDet':       maxDet,
                   'dtIds':        [d['id'] for d in dt],
                   'gtIds':        [g['id'] for g in gt],
                   'dtMatches':    dtm,
                   'gtMatches':    gtm,
                   'dtScores':     [d['score'] for d in dt],
                   'gtIgnore':     gtIg,
                   'dtIgnore':     dtIg,
               }
   ```

   ```c++
   void bbIou( BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o ) {
     double h, w, i, u, ga, da; siz g, d; int crowd;
     for( g=0; g<n; g++ ) {
       BB G=gt+g*4; ga=G[2]*G[3]; crowd=iscrowd!=NULL && iscrowd[g];
       for( d=0; d<m; d++ ) {
         BB D=dt+d*4; da=D[2]*D[3]; o[g*m+d]=0;
         w=fmin(D[2]+D[0],G[2]+G[0])-fmax(D[0],G[0]); if(w<=0) continue;
         h=fmin(D[3]+D[1],G[3]+G[1])-fmax(D[1],G[1]); if(h<=0) continue;
         i=w*h; u = crowd ? da : da+ga-i; o[g*m+d]=i/u;
       }
     }
   }
   ```

## accumulate the individual evaluation. 
   The basic logic is 1) to calculate the recall first, 2) split the recall from 0 to 1 with a step of 0.01, 3) find the
precision at each recall, and then 4) store the results into a variable.
   ```python
   class COCOeval:
       def accumulate(self, p = None):
           '''
           Accumulate per image evaluation results and store the result in self.eval
           :param p: input params for evaluation
           :return: None
           '''
           print('Accumulating evaluation results...')
           tic = time.time()
           if not self.evalImgs:
               print('Please run evaluate() first')
           # allows input customized parameters
           if p is None:
               p = self.params
           p.catIds = p.catIds if p.useCats == 1 else [-1]
           T           = len(p.iouThrs)
           R           = len(p.recThrs)
           K           = len(p.catIds) if p.useCats else 1
           A           = len(p.areaRng)
           M           = len(p.maxDets)
           precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
           recall      = -np.ones((T,K,A,M))
           scores      = -np.ones((T,R,K,A,M))
   
           # create dictionary for future indexing
           _pe = self._paramsEval
           catIds = _pe.catIds if _pe.useCats else [-1]
           setK = set(catIds)
           setA = set(map(tuple, _pe.areaRng))
           setM = set(_pe.maxDets)
           setI = set(_pe.imgIds)
           # get inds to evaluate
           k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
           m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
           a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
           i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
           I0 = len(_pe.imgIds)
           A0 = len(_pe.areaRng)
           # retrieve E at each category, area range, and max number of detections
           for k, k0 in enumerate(k_list):
               Nk = k0*A0*I0
               for a, a0 in enumerate(a_list):
                   Na = a0*I0
                   for m, maxDet in enumerate(m_list):
                       E = [self.evalImgs[Nk + Na + i] for i in i_list]
                       E = [e for e in E if not e is None]
                       if len(E) == 0:
                           continue
                       dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
   
                       # different sorting method generates slightly different results.
                       # mergesort is used to be consistent as Matlab implementation.
                       inds = np.argsort(-dtScores, kind='mergesort')
                       dtScoresSorted = dtScores[inds]
   
                       dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                       dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                       gtIg = np.concatenate([e['gtIgnore'] for e in E])
                       npig = np.count_nonzero(gtIg==0 )
                       if npig == 0:
                           continue
                       tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                       fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
   
                       tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                       fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                       for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                           tp = np.array(tp)
                           fp = np.array(fp)
                           nd = len(tp)
                           rc = tp / npig
                           pr = tp / (fp+tp+np.spacing(1))
                           q  = np.zeros((R,))
                           ss = np.zeros((R,))
   
                           if nd:
                               recall[t,k,a,m] = rc[-1]
                           else:
                               recall[t,k,a,m] = 0
   
                           # numpy is slow without cython optimization for accessing elements
                           # use python array gets significant speed improvement
                           pr = pr.tolist(); q = q.tolist()
   
                           for i in range(nd-1, 0, -1):
                               if pr[i] > pr[i-1]:
                                   pr[i-1] = pr[i]
   
                           inds = np.searchsorted(rc, p.recThrs, side='left')
                           try:
                               for ri, pi in enumerate(inds):
                                   q[ri] = pr[pi]
                                   ss[ri] = dtScoresSorted[pi]
                           except:
                               pass
                           precision[t,:,k,a,m] = np.array(q)
                           scores[t,:,k,a,m] = np.array(ss)
           self.eval = {
               'params': p,
               'counts': [T, R, K, A, M],
               'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               'precision': precision,
               'recall':   recall,
               'scores': scores,
           }
           toc = time.time()
           print('DONE (t={:0.2f}s).'.format( toc-tic))
   ```

## the last is to summarize the result and print the number

```python
class COCOeval:
    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        iouType = self.params.iouType
        summarize = _summarizeDets
        self.stats = summarize()
```
