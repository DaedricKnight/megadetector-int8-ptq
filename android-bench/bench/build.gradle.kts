plugins {
    id("com.android.application")   // AGP 9 compiles Kotlin natively
}

android {
    namespace = "com.megadetector.bench"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.megadetector.bench"
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // Real phones & cloud farms are arm64 — drop the other ABIs' ONNX libs
        // so the APK is small enough to upload comfortably.
        ndk { abiFilters += "arm64-v8a" }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // ONNX Runtime mmaps the model from disk; compressed assets can't be mmap'd.
    androidResources { noCompress += "onnx" }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
