Projeto de Redes Neurais - CIn/UFPE - 2016.1    
Classificador para detecção de cancer de mama baseado numa base pre-processada, porém desbalanceada    
    
Instalação:    

Windows:
- Download do python: https://www.python.org/downloads/
- Download do pip: https://pypi.python.org/pypi/pip#downloads
- Instalar o numpy + MLK para a versão especifica do python e sistema pelo link: http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
- Instalar o scipy
- Instalar o scikit-learn
- Instalar o UnbalancedDataset:
```powershell
$ pip install git+https://github.com/fmfn/UnbalancedDataset
```    
- Instalar o ipdb    

Debian/Ubuntu distros:
```powershell
$ sudo apt-get install -y python3.5
$ sudo apt-get install -y pip
$ sudo pip install numpy
$ sudo pip install scipy
$ sudo pip install scikit-learn
$ sudo pip install git+https://github.com/fmfn/UnbalancedDataset
$ sudo pip install ipdb
```    

Rodando:

```powershell
$ python main.py
```