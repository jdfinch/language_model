
# Language Model

## Install

```shell
git clone https://github.com/jdfinch/language_model.git
cd language_model
git clone https://github.com/jdfinch/ezpyzy.git
cd src
ln -s ../ezpyzy/ezpyzy ezpyzy
cd ..
conda env create --name PROJ python=3.10
conda activate PROJ
pip install -r requirements.txt
```