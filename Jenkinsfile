pipeline {
    agent {
        docker {
            image 'pytorch/pytorch'
            args '--runtime=nvidia -u root -v $HOME/referit_data:/root/referit_data -v $HOME/data:/root/data'
        }
    }

    environment {
        CODECOV_TOKEN = 'afac3345-89ec-41cd-b9ce-17206e10a585'
    }

    stages {
        stage('Build') {
            steps {
                sh 'pip install opencv-python pydocstyle cupy-cuda91'
                sh 'pip install pytest pytest-cov flaky codecov pytest-xvfb pytest-timeout'
                sh 'conda install mccabe flake8 pycodestyle'
                sh 'conda install aria2 -c bioconda'
                sh 'git clone https://github.com/taolei87/sru.git'
                sh 'cd sru'
                sh 'ls'
                sh 'pip install -U . --no-deps'
                sh 'cd ..'
                sh 'python -c "from sru import SRU"'
                sh 'bash download_data.sh -p $HOME/referit_data'
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
