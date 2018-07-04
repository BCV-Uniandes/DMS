pipeline {
    agent none
    stages {
        stage('Build') {
            agent {
                docker {
                    image 'pytorch/pytorch'
                    args '--runtime=nvidia -u root'
                }
            }
            steps {
                sh 'nvidia-smi'
                sh 'pip install opencv-python pydocstyle cupy-cuda91'
                sh 'conda install mccabe flake8 pycodestyle'
                sh 'flake8 --exclude=*/optimizers/*,*/dpn/* --max-complexity 16 .'
            }
        }
    }
}
