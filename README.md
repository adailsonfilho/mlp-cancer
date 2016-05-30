#### Projeto de Redes Neurais - CIn/UFPE - 2016.1    
##### Classificador para detecção de cancer de mama baseado numa base pre-processada, porém desbalanceada    

**Instalação:**    

*Windows:*
- Download do python: https://www.python.org/downloads/
- Download do pip: https://pypi.python.org/pypi/pip#downloads
- Instalar o numpy + MLK para a versão especifica do python e sistema pelo link: http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
- Instalar o scipy
- Instalar o scikit-learn
- Instalar o scikit-neuralnetwork
- Instalar o theano: http://rosinality.ncity.net/doku.php?id=python:installing_theano
- Instalar o UnbalancedDataset:
```powershell
$ pip install git+https://github.com/fmfn/UnbalancedDataset
```    
- Instalar o ipdb
- Instalar o matplotlib

*Debian/Ubuntu:*
```powershell
$ sudo apt-get install -y python3.5 python3.5dev
$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.5 1
$ sudo update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
$ sudo apt-get install -y libblas-dev liblapack-dev libatlas-base-dev gfortran
$ sudo apt-get install -y pip
$ sudo pip install -Iv numpy==1.7.0
$ sudo pip install scipy
$ sudo pip install scikit-learn
$ sudo pip install scikit-neuralnetwork
$ sudo pip install git+https://github.com/fmfn/UnbalancedDataset
$ sudo pip install ipdb
$ sudo pip install matplotlib
```    

Rodando:

```powershell
$ python main.py
```
