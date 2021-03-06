---
layout: post
comments: true
title: MongoDB Pipeline Examples
---

* How to visualize the UHRS verification result
  ```python
  [{'$match': {'interpretation_result': 1, 'status': 'completed'}},
   {'$addFields': {'gt': '$rects', 'key': {'$concat': ['$data', '_', '$key']}}}]
  ```

* How to compare our prediction results with Google's 
    * Google's MIT results are saved as the 6th version
    * Our results are based on full_expid and predict_file

      ``` python
      [{'$match': {'data': 'MIT1K-GUID', 'split': 'test'}},
       {'$skip': 0},
       {'$limit': 200},
       {'$lookup': {'as': 'gt',
                    'from': 'ground_truth',
                    'let': {'data': '$data', 'key': '$key', 'split': '$split'},
                    'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data',
                                                                         '$$data']},
                                                                {'$eq': ['$split',
                                                                         '$$split']},
                                                                {'$eq': ['$key',
                                                                         '$$key']},
                                                                {'$lte': ['$version',
                                                                          6]}]}}},
                                 {'$group': {'_id': {'action_target_id': '$action_target_id'},
                                             'class': {'$first': '$class'},
                                             'conf': {'$first': '$conf'},
                                             'contribution': {'$sum': '$contribution'},
                                             'rect': {'$first': '$rect'}}},
                                 {'$match': {'contribution': {'$gte': 1}}}]}},
       {'$lookup': {'as': 'pred',
                    'from': 'predict_result',
                    'let': {'data': '$data', 'key': '$key', 'split': '$split'},
                    'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$full_expid',
                                                                         'Tax1300V14.2_0.0_0.1_darknet19_448_C_Init.best_model6933_maxIter.4.5eEffectBatchSize128LR7580_bb_only']},
                                                                {'$eq': ['$pred_file',
                                                                         'model_iter_71333.caffemodel.MIT1K-GUID.test.maintainRatio.OutTreePath.TreeThreshold0.1.ClsIndependentNMS.predict']},
                                                                {'$eq': ['$key',
                                                                         '$$key']},
                                                                {'$gte': ['$conf',
                                                                          0.6]}]}}},
                                 {'$project': {'_id': 0,
                                               'class': True,
                                               'conf': True,
                                               'rect': True}}]}},
       {'$project': {'bbs': {'$map': {'as': 'x',
                                      'in': {'$cond': [{'$eq': ['$$x', 'gt']},
                                                       '$gt',
                                                       '$pred']},
                                      'input': {'$literal': ['gt', 'pred']}}},
                     'key': True,
                     'url': True}},
       {'$unwind': {'includeArrayIndex': 'idx',
                    'path': '$bbs',
                    'preserveNullAndEmptyArrays': True}},
       {'$addFields': {'gt': {'$cond': [{'$eq': ['$idx', 0]}, '$bbs', []]},
                       'key': {'$cond': [{'$eq': ['$idx', 0]},
                                         {'$concat': ['$key', '_Google']},
                                         {'$concat': ['$key', '_MS']}]},
                       'pred': {'$cond': [{'$eq': ['$idx', 0]}, [], '$bbs']}}}]
      ```

* How to query all the images from PPBDataset whose label is Sweden, but we have not predicted it as person
    * The ground-truth label is kept in version 0. 
    * The prediction results from model of Tax1300V11_3 is in version 1
    * We use the label in version 0 as the field of gt, and the labels in version 1 as pred
      ``` python
      [{'$match': {
              'data': 'PPBDataset', 
              'split': 'test', 
              'class': 'Sweden', 
              }},
      {'$group': {'_id': {'key': '$key'}}},
      ## add url
      {'$lookup': {   
          'from': 'image', 
          'let': {'key': '$_id.key'},
          'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data', 'PPBDataset']},
                                                      {'$eq': ['$split', 'test']},
                                                      {'$eq': ['$key', '$$key']},],}}},
                       {'$project': {'url': True, '_id': 0}}],
          'as': 'url',
          }},
      {'$addFields': {'url': {'$arrayElemAt': ['$url', 0]}}},
      {'$addFields': {'url': '$url.url'}},
      {'$lookup': {
          'from': 'ground_truth',
          'let': {'key': '$_id.key'},
          'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data', 'PPBDataset']},
                                                      {'$eq': ['$split', 'test']},
                                                      {'$eq': ['$key', '$$key']},
                                                      {'$lte':['$version', 0]}],}}},
                      {'$group': {'_id': {'action_target_id': '$action_target_id'},
                                  'contribution': {'$sum': '$contribution'},
                                  'rect': {'$first': '$rect'},
                                  'class': {'$first': '$class'},
                                  'conf': {'$first': '$conf'}}, },
                      {'$match': {'contribution': {'$gte': 1}}},
                      ],
          'as': 'gt',
          }},
      {'$lookup': {
          'from': 'ground_truth',
          'let': {'key': '$_id.key'},
          'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data', 'PPBDataset']},
                                                      {'$eq': ['$split', 'test']},
                                                      {'$eq': ['$key', '$$key']},
                                                      {'$lte':['$version', 1]}],}}},
                      {'$group': {'_id': {'action_target_id': '$action_target_id'},
                                  'contribution': {'$sum': '$contribution'},
                                  'rect': {'$first': '$rect'},
                                  'class': {'$first': '$class'},
                                  'conf': {'$first': '$conf'}}, },
                      {'$match': {'contribution': {'$gte': 1}}},
                      ],
          'as': 'pred',
          }},
      {'$unwind': {'path': '$pred', 'preserveNullAndEmptyArrays': True}},
      {'$addFields': {'has_person': {'$eq': ['$pred.class', 'person']}}},
      {'$group': {'_id': {'key': '$_id.key'},
                  'gt': {'$first': '$gt'},
                  'url': {'$first': '$url'},
                  'pred': {'$push': '$pred'},
                  'num_person': {'$sum': {'$cond': ['$has_person', 1, 0]}}}},
       {'$sort': {'num_person': 1}},
      ]
    ```

* Query images whose ground truth labels has Female, but our prediction has Male.
  ``` python
  {'$match': {'class': 'Female', 'data': 'ExtendedGenderBenchmark', 'split': 'test'}},
  {'$group': {'_id': {'key': '$key'}}},
  {'$lookup': {'as': 'pred',
               'from': 'ground_truth',
               'let': {'key': '$_id.key'},
               'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data',
                                                                    'ExtendedGenderBenchmark']},
                                                           {'$eq': ['$split',
                                                                    'test']},
                                                           {'$eq': ['$key',
                                                                    '$$key']},
                                                           {'$lte': ['$version',
                                                                     1]}]}}},
                            {'$group': {'_id': {'action_target_id': '$action_target_id'},
                                        'class': {'$first': '$class'},
                                        'conf': {'$first': '$conf'},
                                        'contribution': {'$sum': '$contribution'},
                                        'rect': {'$first': '$rect'}}},
                            {'$match': {'contribution': {'$gte': 1}}}]}},
  {'$unwind': {'path': '$pred', 'preserveNullAndEmptyArrays': True}},
  {'$addFields': {'has_male': {'$eq': ['$pred.class', 'male']}}},
  {'$group': {'_id': {'key': '$_id.key'},
              'num_male': {'$sum': {'$cond': ['$has_male', 1, 0]}},
              'pred': {'$push': '$pred'}}},
  {'$match': {'num_male': {'$gte': 1}}},
  {'$sort': {'num_male': -1}},
  {'$limit': 1000},
  {'$lookup': {'as': 'url',
               'from': 'image',
               'let': {'key': '$_id.key'},
               'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data',
                                                                    'ExtendedGenderBenchmark']},
                                                           {'$eq': ['$split',
                                                                    'test']},
                                                           {'$eq': ['$key',
                                                                    '$$key']}]}}},
                            {'$project': {'_id': 0, 'url': True}}]}},
  {'$addFields': {'url': {'$arrayElemAt': ['$url', 0]}}},
  {'$addFields': {'url': '$url.url'}},
  {'$lookup': {'as': 'gt',
               'from': 'ground_truth',
               'let': {'key': '$_id.key'},
               'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data',
                                                                    'ExtendedGenderBenchmark']},
                                                           {'$eq': ['$split',
                                                                    'test']},
                                                           {'$eq': ['$key',
                                                                    '$$key']},
                                                           {'$lte': ['$version',
                                                                     0]}]}}},
                            {'$group': {'_id': {'action_target_id': '$action_target_id'},
                                        'class': {'$first': '$class'},
                                        'conf': {'$first': '$conf'},
                                        'contribution': {'$sum': '$contribution'},
                                        'rect': {'$first': '$rect'}}},
                            {'$match': {'contribution': {'$gte': 1}}}]}},
  
  ]
  ```
