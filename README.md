# Deep Beyond Android version

### 環境
* Windows 10
* Android Studio
* Java JDK Version 18
* Compile Sdk Version 33

### ファイル構造
```
C:.
│  .gitignore
│  build.gradle.kts
│  check_model_IO.py    <--- TF Liteモデルの入出力のサイズを確認するため
│  filetree.txt
│  gradle.properties
│  gradlew
│  gradlew.bat
│  local.properties
│  README.md
│  settings.gradle.kts
│              
├─app
│  │  .gitignore
│  │  build.gradle.kts
│  │  proguard-rules.pro
│  │          
│  └─src
│      │                      
│      ├─main
│      │  │  AndroidManifest.xml
│      │  │  
│      │  ├─assets
│      │  │      lite-model_mobilenetv2-dm05-coco_dr_1.tflite
│      │  │      
│      │  ├─java
│      │  │  └─com
│      │  │      └─example
│      │  │          └─deepbeyond
│      │  │              │  MainActivity.kt
│      │  │              │  Segmantation.kt
│      │  │              │  
│      │  │              └─ui
│      │  │                  └─theme
│      │  │                          Color.kt
│      │  │                          Theme.kt
│      │  │                          Type.kt
│      │  │                          
│      │  ├─jniLibs
│      │  │  ├─arm64-v8a
│      │  │  │      libopencv_java4.so
│      │  │  │      
│      │  │  ├─armeabi-v7a
│      │  │  │      libopencv_java4.so
│      │  │  │      
│      │  │  ├─x86
│      │  │  │      libopencv_java4.so
│      │  │  │      
│      │  │  └─x86_64
│      │  │          libopencv_java4.so
│ 
│          
└─sdk
    │  build.gradle
    │  local.properties
```