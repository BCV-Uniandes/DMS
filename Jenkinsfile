pipeline {
    agent {
        docker {
            image 'pytorch/pytorch'
            args '--runtime=nvidia -u root'
        }
    }
    stages {
        stage('Build') {
            steps {
                sh 'pip install opencv-python pydocstyle cupy-cuda91'
                sh 'conda install mccabe flake8 pycodestyle'
                sh 'git clone https://github.com/taolei87/sru.git'
                sh 'cd sru && pip install -U . --no-deps && cd .. & rm -rf sru'
            }
        }
        stage('Run Tests') {
          parallel {
            stage('Code Linting') {
              steps {
                  sh 'flake8 --exclude=*/optimizers/*,*/dpn/* --max-complexity 16 .'
              }
            }
            stage('Model Training') {
              steps {
                sh 'nvidia-smi'
              }
            }
          }
        }
    }
}
