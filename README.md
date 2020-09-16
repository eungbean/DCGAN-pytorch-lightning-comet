
<!-- ## Deep Learning Clean Template For A.I. Researcher -->

## DCGAN in 100-Days-of-GAN
---
[![Comet.ML: experiments](https://img.shields.io/badge/Comet.ml-experiments-orange.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAhpJREFUOI2Nk01IVGEUhp/zzfW3sUgllUIEzVApS2JyIwStylYmlC3EWvWzcNGyhdAqiGpZy4ja2UYMrBgoatnCTCNRJBxBpNE0bcbx3rlviyycScF3d+B93nMO33eMPEkiRLEs9BRgdYDzUSKCDTl4bWY5/pwqlJoNBkllmjZevWVjYgrzA37V13LgwjnYUzwH9JlZPL8xodQlSWsPHynJES3VnFTiYLvelcQ0QptGiEl+Vpvqz5kglBoMplZvXsP/8J5I+X5WffiWFMpECANHai5Bx2ScaGPdX/asmY14kgAG088eE0y9IdJYSSif+ek03uJP/KQjRJR3tG+FAZ5IqrZAaowQTK5c2gv7GjEzFhPfsdPXqbnchzmPTHKJsqNN/60NdHtBSK99jkNlGoumMGUoPXaFqhu3/7kKa6q2gwHOuyJHbbY0pOjiLbyWNojOUtHVuxOQr1oHECy8xJ+5T6S1haLOu7gSt9sAXChmS07dofDEAwoarpJd/YorrdwtP+s2Ap6yNIqLHiKYeYEyK6yP3gPC3QQM2+Yzjq5/HGj1Ko4TqahHmR+4gkKIHgZvx2kWgGpnZqQDuovrO7HiMswV4pbjkJ2H9Bikxzf92fyAvpy7yAR/vrKmB6TUuLQ8LCWfS6kxKTUh+Qvaov78NAAkNUv6IklaiUtrn6RsaiuYkHRmK2PbhADE/ICeAo86wAEJYIhtzvk3y+cYpafNe/QAAAAASUVORK5CYII=)](https://www.comet.ml/eungbean/dcgan/4b5175430fa0445aa656516bbfa77fe5?experiment-tab=panels)

> written by Eungbean Lee  
> 이 프로젝트는 연세대학교 어영정 교수님의 '[GAN (AAI5007)](http://ysweb.yonsei.ac.kr:8888/curri120601/curri_pop2.jsp?&hakno=AAI5007&bb=01&sbb=00&domain=A&startyy=2020&hakgi=2&ohak=10750)'수업과 함께 진행되었습니다.

* Implemantation of DCGAN written in pytorch
* Super-clean and easy-configurable code equipped with variety state-of-arts high-level tools for deep learning.
    - `Docker`
    - `Pytorch-lightning`, The lightweight PyTorch wrapper for high-performance AI research.
    - `Comet.ml` provides a self-hosted and cloud-based meta ML platform allowing to track and optimize experiments and models.
    - `YACS` is designed to be simple configuration management system for academic and industrial research projects.
    - `Albumentation` is a fast image augmentation library and easy to use wrapper around other libraries.




이 프로젝트는 Pytorch를 기반으로 바닥부터 구현하였습니다.
추후 Boilerplate Template으로 Refactor 후 배포될 예정입니다.

본 코드는 아래의 특징을 모토로 설계하였습니다. 

- `Dev Productivity`: - 어떤 환경에서도 5분 안에 실행할 수 있습니다. Docker를 사용해 더이상 "안돌아가요!!"라는 issue를 듣지 않아도 됩니다.
- `Clean Coding`: 단순함과 직관성. [Clean Coding Rule](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)을 준수하려고 노력하였습니다.
- `modularity`: 각자의 기능이 다른 조각들을 서로 다른 파이썬 sub module로 분리하였습니다.
- `data-augmentation`: [Albumentation](https://github.com/albumentations-team/albumentations) 패키지를 포함했습니다.
- `ready to go`: 혁신적인 [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) 라이브러리를 사용해 train loop을 쓰지 않아도 됩니다.
- [Comet.ml](https://www.comet.ml/) 을 내장했습니다. ID만 입력하면 모든 학습현황, 파라메터를 실시간으로 모니터링 할 수 있습니다.
- python [logging](https://docs.python.org/3/library/logging.html) module을 사용해 logging을 할 수 있어 SSH등의 상황에서 터미널 연결이 끊겨도 추적이 가능합니다.
- a playground notebook 역시 포함해 적용해 빠른 코딩에 앞서 테스트 후 적용이 가능합니다.

---

이 프로젝트를 구현하는데 도움을 받은 `Github Project`/`Blog Post`는 다음과 같습니다.
- [clean-code-ml](https://github.com/davified/clean-code-ml) by David Tan
- [Coding habits for data scientists](https://www.thoughtworks.com/insights/blog/coding-habits-data-scientists) by David Tan

- [Pytorch Deep Learning Template](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Template/tree/master) by Francesco Saverio Zuppichini [Blog](https://towardsdatascience.com/pytorch-deep-learning-template-6e638fc2fe64)
- [Best practices to write Deep Learning code: Project structure, OOP, Type checking and documentation](https://theaisummer.com/best-practices-deep-learning-code/) by Sergios Karagiannakos 
- [Tips for Publishing Research Code](https://github.com/paperswithcode/releasing-research-code) by paperswithcode
- [AI 연구자를 위한 클린코드](https://www.slideshare.net/KennethCeyer/ai-gdg-devfest-seoul-2019-187630418) by 한성민

## Structure
```
.
├── callbacks // here you can create your custom callbacks
├── checkpoint // were we store the trained models
├── data // here we define our dataset
│ └── transformation // custom transformation, e.g. resize and data augmentation
├── dataset // the data
│ ├── train
│ └── val
├── logger.py // were we define our logger
├── losses // custom losses
├── main.py
├── models // here we create our models
│ ├── MyCNN.py
│ ├── resnet.py
│ └── utils.py
├── playground.ipynb // a notebook that can be used to fast experiment with things
├── Project.py // a class that represents the project structure
├── README.md
├── requirements.txt
├── test // you should always perform some basic testing
│ └── test_myDataset.py
└── tools.utils.py // utilities functions
```