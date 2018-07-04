pipeline {
    agent none
    stages {
        stage('Build') {
            agent {
                docker {
                    image 'pytorch/pytorch'
                    args '--runtime=nvidia'
                }
            }
            steps {
                sh 'nvidia-smi'
            }
        }
    }
}
