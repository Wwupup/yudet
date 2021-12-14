pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
opencv
pip install opencv-python
nvidia-dali
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
scipy
conda install scipy
tqdm
conda install tqdm
jupyterlab
conda install -c conda-forge jupyterlab

Cython
conda install Cython
cd tools/widerface_eval
python setup.py build_ext --inplace

tensorboard
conda install tensorboard

yaml
pip install pyyaml

widerface_eval
python setup.py build_ext --inplace



//
+ log
++ 1. 2gpu2 imagenet\
++ 2. 2gpu_0 doublehead\
++ 3. 2gpu_1 pan