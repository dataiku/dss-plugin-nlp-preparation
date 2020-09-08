pipeline {
   agent { label 'dss-plugin-tests'}
   environment {
        HOST = "$dss_target_host"
    }
   stages {
      stage('Run Unit Tests') {
         steps {
            sh 'echo "Running unit tests"'
            sh """
               make unit-tests
               """
            sh 'echo "Done with unit tests"'
         }
      }
//       stage('Run Integration Tests') {
//          steps {
//              withCredentials([string(credentialsId: 'dss-plugins-admin-api-key', variable: 'API_KEY')]) {
//                 sh 'echo "Running integration tests"'
//                 sh 'echo "$HOST"'
//                 sh """
//                    make integration-tests
//                    """
//                 sh 'echo "Done with integration tests"'
//              }
//          }
//       }
    }
   post {
     always {
        junit '*.xml'
     }
   }
}
