![Image](/images/mmsys-logo.png)

This code is developed as a part of [ACM MMSys Grand Challenge on 360 degree video streaming](https://2024.acmmmsys.org/gc/360-vod/). 

# Quickstart
* Clone the latest [E3P0](https://github.com/bytedance/E3PO/tree/main) code
* Follow instructions and implement E3PO
* Create a folder called `360LCY` inside `e3p0/approaches`
* Download all source files from this repo and place them inside `e3p0/approaches/360LCY`
* Weights for viewport prediction can be found [here](https://kingstonuniversity-my.sharepoint.com/:u:/g/personal/ku75509_kingston_ac_uk/EUvtM0ciscdLnv__MAjHtaUBOKRJlBLkg7nWNpiZRB-0Rg?e=C6PmdV)
* Place the weights inside `e3p0/approaches/360LCY/viewport/model`

# Running the software
* The configurations relating to tiling can be modified using `e3p0/approaches/360LCY/360LCY.yaml`
* To run the preprocessing module



```python ./e3po/make_preprocessing.py -approach_name erp -approach_type on_demand```


# ACKOWLEDGMENT

