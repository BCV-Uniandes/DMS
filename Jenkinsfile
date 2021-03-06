pipeline {
    agent {
        docker {
            image 'pytorch/pytorch:0.4-cuda9-cudnn7-devel'
            args '--runtime=nvidia -u root -v /media/SSD2/referit_data:/root/referit_data -v /media/SSD2/test_data:/root/data -v /home/eamargffoy/.torch:/root/data/.torch -v /home/eamargffoy/.cache:/root/data/.cache -e PYTHONIOENCODING=UTF-8'
        }
    }

    environment {
        CODECOV_TOKEN = 'afac3345-89ec-41cd-b9ce-17206e10a585'
    }

    stages {
        stage('Build') {
            steps {
                sh 'apt-get update'
                sh 'apt-get install libgtk2.0-dev -y'
                sh 'pip install opencv-python pydocstyle cupy-cuda90 pynvrtc'
                sh 'pip install pytest pytest-cov flaky codecov pytest-xvfb pytest-timeout progressbar2'
                sh 'conda install mccabe flake8 pycodestyle cython'
                sh 'pip install --no-deps git+https://github.com/andfoy/sru.git@encoding'
                sh 'pip install git+https://github.com/andfoy/refer.git'
                sh 'python -c "from sru import SRU"'
            }
        }
        stage('Run Tests') {
          parallel {
            stage('Code Linting') {
              steps {
                  sh 'flake8 --exclude=*/optimizers/*,*/dpn/*,*/tests/*,*__init__* --max-complexity 16 .'
              }
            }
            stage('Model Forward Pass') {
              steps {
                sh 'nvidia-smi'
                sh 'ls /root/referit_data'
                sh 'pytest dmn_pytorch --cov=dmn_pytorch --cov-report term-missing -v -p no:cacheprovider --cache-clear'
                sh 'codecov'
                sh 'find . -regex \'.*\\(__pycache__\\|\\.py[co]\\)\' -delete'
              }
            }
          }
        }
    }
}
