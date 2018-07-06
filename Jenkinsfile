pipeline {
    agent {
        docker {
            image 'pytorch/pytorch:0.4-cuda9-cudnn7-devel'
            args '--runtime=nvidia -u root -v $HOME/referit_data:/root/referit_data -e PYTHONIOENCODING=UTF-8'
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
                sh 'pip install pytest pytest-cov flaky codecov pytest-xvfb pytest-timeout'
                sh 'conda install mccabe flake8 pycodestyle'
                sh 'conda install aria2 -c bioconda'
                sh 'export PYTHONIOENCODING=utf-8 && pip install --no-deps git+https://github.com/andfoy/sru.git@encoding'
                sh 'python -c "from sru import SRU"'
            }
        }
        stage('Run Tests') {
          parallel {
            stage('Code Linting') {
              steps {
                  sh 'flake8 --exclude=*/optimizers/*,*/dpn/*,*/test/* --max-complexity 16 .'
              }
            }
            stage('Model Forward Pass') {
              steps {
                sh 'nvidia-smi'
                sh 'pytest dmn_pytorch --cov=dmn_pytorch --cov-report term-missing -v -p no:cacheprovider'
                sh 'codecov'
              }
            }
          }
        }
    }
}
