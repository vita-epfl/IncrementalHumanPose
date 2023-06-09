# Incremental Learning with Human Pose
Kardelen Ceren
CS398 – Project in Computer Science I, 8 ECTS, Spring 2023

EPFL, Visual Intelligence for Transportation (VITA) Laboratory

Professor: Alexandre Alahi, Supervisor: Megh Shukla

Code built upon https://github.com/meghshukla/ActiveLearningForHumanPose/tree/main

This project aims to develop an incremental learning approach for human pose estimation. Traditional models struggle to adapt to new poses without extensive retraining. To overcome this challenge, we explore incremental learning techniques that enable models to learn from new data while retaining previous knowledge, reducing catastrophic forgetting. We modify the Stacked Hourglass model, incorporate task heads, and investigate regularization methods. By comparing different incremental learning strategies to baselines on the MPII and LSP datasets, we highlight the importance of different incremental learning strategies in improving human pose estimation algorithms.

## Datasets
 MPII dataset can be found [here](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz). LSP can be found [here](http://web.archive.org/web/20220426074820/http://sam.johnson.io/research/lsp.html).
 After downloading, add their "images" folder to this repository's data folder, under appropriate name. 