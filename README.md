# Point cloud classification based on pointnet++ and textcnn

## Log
- 2018.9.18 finish textcnn_train.py and model_cls.py
- 2018.9.19 firstly run on server; finish textcnn_evaluate.py
- 2018.9.20 second run on server finished(takes one night and one day and it crashed in epoch 180)
- 2018.9.21 third run manually quit; perform worse than the second time

## History results
- 2018.9.19
```
hyperparameters:
  N: 1024 (total pointcloud number)
  M: 128 (sampled point number)
  Ki: [16, 32, 64] (number of knn points)
  textcnn_kernel: [3] * 128
  epochs: 251
  other: remain the same as pointnet2
```
```
eval mean loss: 0.396773
eval accuracy: 0.882496
eval avg class acc: 0.846058
  airplane:	1.000
   bathtub:	0.840
       bed:	0.950
     bench:	0.700
 bookshelf:	0.890
    bottle:	0.960
      bowl:	1.000
       car:	1.000
     chair:	0.970
      cone:	0.950
       cup:	0.800
   curtain:	0.800
      desk:	0.826
      door:	0.950
   dresser:	0.756
flower_pot:	0.000
 glass_box:	0.850
    guitar:	0.990
  keyboard:	1.000
      lamp:	0.850
    laptop:	1.000
    mantel:	0.940
   monitor:	0.970
night_stand:	0.721
    person:	0.900
     piano:	0.920
     plant:	0.830
     radio:	0.700
range_hood:	0.890
      sink:	0.650
      sofa:	0.950
    stairs:	0.900
     stool:	0.800
     table:	0.860
      tent:	0.950
    toilet:	0.980
  tv_stand:	0.820
      vase:	0.780
  wardrobe:	0.500
      xbox:	0.700
```
- 2018.09.20
```
hyperparameters:
  N: 1024 (total pointcloud number)
  M: 384 (sampled point number)
  Ki: [16, 32, 64, 128] (number of knn points)
  textcnn_kernel: [1,2,3,4] * 32
  epochs: 180
  other: remain the same as pointnet2
```
```
eval mean loss: 0.385354
eval accuracy: 0.895462
eval avg class acc: 0.864849
  airplane: 1.000
   bathtub: 0.820
       bed: 0.990
     bench: 0.750
 bookshelf: 0.910
    bottle: 0.950
      bowl: 0.950
       car: 0.990
     chair: 0.960
      cone: 0.900
       cup: 0.800
   curtain: 0.800
      desk: 0.814
      door: 0.900
   dresser: 0.709
flower_pot: 0.200
 glass_box: 0.940
    guitar: 0.990
  keyboard: 1.000
      lamp: 0.850
    laptop: 1.000
    mantel: 0.940
   monitor: 0.970
night_stand:  0.791
    person: 0.950
     piano: 0.920
     plant: 0.800
     radio: 0.800
range_hood: 0.940
      sink: 0.700
      sofa: 0.940
    stairs: 0.900
     stool: 0.850
     table: 0.890
      tent: 1.000
    toilet: 0.990
  tv_stand: 0.860
      vase: 0.780
  wardrobe: 0.600
      xbox: 0.750
```
- 2018.09.21
```
hyperparameters:
  N: 1024 (total pointcloud number)
  M: 384 (sampled point number)
  Ki: [16, 32, 64, 128] (number of knn points)
  textcnn_kernel: [1,2,3,4] * 64
  epochs: 161
  other: remain the same as pointnet2
```
```
eval mean loss: 0.402381
eval accuracy: 0.887763
eval avg class acc: 0.857355
  airplane: 1.000
   bathtub: 0.860
       bed: 0.980
     bench: 0.700
 bookshelf: 0.920
    bottle: 0.940
      bowl: 0.950
       car: 1.000
     chair: 0.970
      cone: 1.000
       cup: 0.750
   curtain: 0.850
      desk: 0.837
      door: 0.900
   dresser: 0.651
flower_pot: 0.200
 glass_box: 0.920
    guitar: 0.990
  keyboard: 1.000
      lamp: 0.900
    laptop: 1.000
    mantel: 0.940
   monitor: 0.970
night_stand:  0.756
    person: 0.850
     piano: 0.910
     plant: 0.800
     radio: 0.850
range_hood: 0.900
      sink: 0.700
      sofa: 0.930
    stairs: 0.900
     stool: 0.850
     table: 0.860
      tent: 0.950
    toilet: 0.990
  tv_stand: 0.800
      vase: 0.820
  wardrobe: 0.400
      xbox: 0.800
```