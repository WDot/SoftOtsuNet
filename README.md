# SoftOtsuNet

This is a Data Augmentation method designed for Dermatology Classification. Our tech report describing it is under review at AMIA 2023.

To run our method, you must have access to the Fitzpatrick17k dataset. First build the Dockerfile:

    docker build --tag=equity:0.1 .
  
Next, edit run.sh to insert the path on your machine to this directory, to the Fitzpatrick17k image directory, and to the Fitzpatrick17k label CSV file. Then run

    bash run.sh
    
We trained each network on 4 NVIDIA Quadro RTX 5000s on a single server.

If you would like to cite us, the bibtex is as follows.

    @inproceedings{
    softotsunet,
    title={Unsupervised SoftOtsuNet Augmentation for Clinical Dermatology Classifieres},
    author={Miguel Dominguez and John T Finnell},
    booktitle={Under review at AMIA Symposium 2023},
    year={2023}
    }
