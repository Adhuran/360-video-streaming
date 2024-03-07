![Image](/images/ERP.png)

This code is developed as a part of [ACM MMSys Grand Challenge on 360 degree video streaming](https://2024.acmmmsys.org/gc/360-vod/). This functionality supports on-demand streaming scenarios of 360-degree videos.

# Quickstart
* Clone the latest [E3P0](https://github.com/bytedance/E3PO/tree/main) code
* Follow instructions and implement E3PO
* Create a folder called `360LCY` inside `e3p0/approaches`
* Download all source files from this repo and place them inside `e3p0/approaches/360LCY`
* Pre-trained weights for viewport prediction can be found [here](https://kingstonuniversity-my.sharepoint.com/:u:/g/personal/ku75509_kingston_ac_uk/EUvtM0ciscdLnv__MAjHtaUBOKRJlBLkg7nWNpiZRB-0Rg?e=C6PmdV)
* Create a folder `e3p0/approaches/360LCY/viewport/model` and save the weights.
* Install the requirements `pip install -r requirements.txt`

# Running the software
* The configurations relating to tiling can be modified using `e3p0/approaches/360LCY/360LCY.yaml`
* To run the preprocessing module: `python ./e3po/make_preprocessing.py -approach_name 360LCY -approach_type on_demand`
* To run the stream selector module: `python ./e3po/make_preprocessing.py -approach_name 360LCY -approach_type on_demand`
* To run the evalution module: `python ./e3po/make_preprocessing.py -approach_name 360LCY -approach_type on_demand`

# Citation
If this work is used, kindly cite our work in your research.

```
Tobe updated

@ARTICLE{9940241,
  author={Adhuran, Jayasingam and Kulupana, Gosala and Fernando, Anil},
  journal={IEEE Access}, 
  title={Deep Learning and Bidirectional Optical Flow Based Viewport Predictions for 360° Video Coding}, 
  year={2022},
  volume={10},
  number={},
  pages={118380-118396},
  keywords={Videos;Encoding;Video coding;Image coding;Quantization (signal);Optimization;Bit rate;Predictive models;360° video;perceptual coding;Regions of Interest;viewport prediction;Versatile Video Coding},
  doi={10.1109/ACCESS.2022.3219861}}
```

# Contact
If there are any inquiries, kindly contact j.adhuran@kingston.ac.uk or m.martini@kingston.ac.uk 

# Acknowledgement
The code follows [E3PO](https://github.com/bytedance/E3PO) logic and imports a few functions. The Spherical convolutions and Salient ResNet in viewport prediction are adopted from [spherical CNNS](https://github.com/jonkhler/s2cnn) and [Salient-Net](https://github.com/AmigoCDT/Salient-Net/tree/master) repositories respectively. The saliency calculation imports the motion_detection module from the [deepgaze](https://github.com/mpatacchiola/deepgaze/tree/master) repository.  


