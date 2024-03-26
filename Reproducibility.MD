## Readme for reproducibility submission of paper

This research was developed as part of ACM MMSys 2024 Grand Challenge on 360 degree video streaming. 

# Source code information
* Repository: [source repo](https://github.com/Adhuran/GC_MMSys_Challenge/tree/master)
* Programming language: Python 3.8 >
* Modules needed: Read the requirements.txt for more info.
* The code is flexible with hardware requirement.The code was evaluated as part of the challenge and flagged no errors. 

# Test data information
* Test data was part of E3PO final evaluation. More information can be found [here](https://github.com/bytedance/E3PO/blob/main/mmsys24gc/release.md).

# Installation
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