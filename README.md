# Precise NGSIM: Deep Spatial-Temporal Embedding for Vehicle Trajectory Reconstruction and Validation
[![DOI](https://zenodo.org/badge/503547507.svg)](https://zenodo.org/badge/latestdoi/503547507)
Abstract
---------------------

- Spatial-temporal Map (STMap)-based methods have shown great potentials to process high-angle videos for vehicle trajectory reconstruction, which can meet the needs of various data-driven modeling and imitation learning applications.  
- In this paper, we developed Spatial-Temporal Deep Embedding (STDE) model that employs parity constraints at both pixel and instance level to generate instance-aware embeddings for vehicle stripe segmentation on STMap. At pixel level, each pixel was encoded with its 8-neighbor pixels at different ranges and this encoding is subsequently used to guide a neural network to learn the embedding mechanism. At the instance level, a discriminative loss function is designed to pull pixels belonging to the same instance closer and separate mean value of different instances far apart in the embedding space. The output of the spatial-temporal affinity is then optimized by the mutex-watershed algorithm to obtain final clustering results. Based on segmentation metrics, our model outperformed five other baselines that have been used for STMap processing and shows robustness under the influence of shadows, static noises, and overlapping. 
- The designed model is applied to process all public NGSIM US-101 videos to generate complete vehicle trajectories, indicating a good scalability and adaptability. Last but not least, strengths of scanline method with STDE and future directions were discussed. 


Overview  
--------------------

<p align="center"><img src="https://github.com/TeRyZh/Spatial-Temporal-Deep-Embedding-for-Vehicle-Trajectory-Reconstruction-from-High-Angle-Video/blob/main/Images/Overview.png" /></p>

Video Data and Scanlines (NGSIM US 101 Data)
-------------------
<p align="center"><img src="https://github.com/TeRyZh/Spatial-Temporal-Deep-Embedding-for-Vehicle-Trajectory-Reconstruction-from-High-Angle-Video/blob/main/Images/US-101%20Scanlines.png" /></p>



Highlights
--------------------
- Pixel-Parity Spatiotemporal Correlation Learning
<p> The spatiotemporal affinity relationship of each pixel is encoded into N - dimensional vector $R=[r_1,r_2,…, r_{N-1}, r_N]$, where N is the 8-neighbor pixels within the adjacent spatiotemporal window at different ranges  </p>
<p align="center"><img src="https://github.com/TeRyZh/Spatial-Temporal-Deep-Embedding-for-Vehicle-Trajectory-Reconstruction-from-High-Angle-Video/blob/main/Images/SpatioTemporal_Adjacent_Encoding.png" width=40% height=40% /></p>

- Instance-Aware Discriminative Learning
<p> Between class average embedding vector should be distant. Pixels belonging to the same instance should be close to the average embedding vector of that class. In this Instance-Level Discriminative Learning, to reduce computational overhead, we only consider the vehicle stripes that share the same time window. </p>
<p align="center"><img src="https://github.com/TeRyZh/Spatial-Temporal-Deep-Embedding-for-Vehicle-Trajectory-Reconstruction-from-High-Angle-Video/blob/main/Images/Instance%20Level%20Embedding.png" width=50% height=50% /></p>

- Feature Pyramid Multiscale Spatial-Temporal Embedding Architecture
<p>  multi-resolution spatiotemporal correlation learning module used for analyzing pixelwise relationship preserved in STMap. This problem is considered as correlational learning to reason the affinity relationships between center pixels and its neighbors. However, it is impossible to calculate the affinity scores for all pixel pairs. In order to measure both the long- and short-term information, we adopted the pyramid multiscale resolution scheme as shown in the following figure. </p>
<p align="center"><img src="https://github.com/TeRyZh/Spatial-Temporal-Deep-Embedding-for-Vehicle-Trajectory-Reconstruction-from-High-Angle-Video/blob/main/Images/Feature%20Pyramid%20Embedding.png" width=50% height=50% /></p>


Trajecotry Output 👉 [***Link for Video Trajectory***](https://drive.google.com/drive/folders/1BXlYHSgIJm-WGje_HwBCDFLnqNAuF7G5?usp=sharing)
--------------------
- Rear Bumper Trajectory for Vehicle Moving Away
<p align="center"><img src="https://github.com/TeRyZh/Spatial-Temporal-Deep-Embedding-for-Vehicle-Trajectory-Reconstruction-from-High-Angle-Video/blob/main/Images/C4L3_P2_Pixel_Trajectory.png" width=50% height=50% /></p>

- Front Bumper Trajectory for Vehicle Moving Towards
<p align="center"><img src="https://github.com/TeRyZh/Spatial-Temporal-Deep-Embedding-for-Vehicle-Trajectory-Reconstruction-from-High-Angle-Video/blob/main/Images/C3L3_P3_Pixel_Trajectory.png" width=50% height=50% /></p>

License
-------
The source code is available only for academic/research purposes (non-commercial).


Contributing
--------
If you found any issues in our model or new dataset please contact: terry.tianya.zhang@gmail.com

Reference
-----------
- Kong, S., & Fowlkes, C. C. (2018). Recurrent pixel embedding for instance grouping. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 9018-9028).
- Gao N, Shan Y, Wang Y, Zhao X, Yu Y, Yang M, Huang K. Ssap: Single-shot instance segmentation with affinity pyramid. In Proceedings of the IEEE/CVF International Conference on Computer Vision 2019 (pp. 642-651).
- Chen, L., Strauch, M. and Merhof, D., 2019, October. Instance segmentation of biomedical images with an object-aware embedding learned with local constraints. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 451-459). Springer, Cham. 
- Huang, W., Deng, S., Chen, C., Fu, X., & Xiong, Z. (2022). Learning to Model Pixel-Embedded Affinity for Homogeneous Instance Segmentation. Proceedings of the AAAI Conference on Artificial Intelligence, 36(1), 1007-1015. https://doi.org/10.1609/aaai.v36i1.19984
- https://github.com/looooongChen/instance_segmentation_with_pixel_embeddings/
- https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping
