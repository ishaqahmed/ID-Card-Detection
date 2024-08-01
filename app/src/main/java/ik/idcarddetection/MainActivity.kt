package ik.idcarddetection

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mlkit.common.model.LocalModel
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.custom.CustomObjectDetectorOptions
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private val executor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) ==
            PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            startCamera()
        } else {
            // Handle the case where the user denies the permission
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(findViewById<PreviewView>(R.id.previewView).surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(executor) { imageProxy ->
                        detectObjects(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e("ishaq Camera", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    @OptIn(ExperimentalGetImage::class)
    private fun detectObjects(imageProxy: ImageProxy) {
        val localModel = LocalModel.Builder()
            .setAssetFilePath("detect_quant_metadata.tflite")
            // or .setAbsoluteFilePath(absolute file path to model file)
            // or .setUri(URI to model file)
            .build()

        // Live detection and tracking
        val customObjectDetectorOptions =
            CustomObjectDetectorOptions.Builder(localModel)
                .setDetectorMode(CustomObjectDetectorOptions.STREAM_MODE)
                .setClassificationConfidenceThreshold(0.5f)
                .setMaxPerObjectLabelCount(1)
                .enableMultipleObjects()
                .enableClassification()
                .build()

        val objectDetector =
            ObjectDetection.getClient(customObjectDetectorOptions)

        val mediaImage = imageProxy.image ?: return
        val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)

        objectDetector.process(image)
            .addOnSuccessListener { detectedObjects ->
                Log.e("ishaq Object Detection", "${detectedObjects.size}")
                for (detectedObject in detectedObjects) {
                    for (label in detectedObject.labels) {
                        Log.e("ishaq Object Detection", "${label.text}")
                        if (label.text == "card" && detectedObject.boundingBox.intersect(Rect(50, 50, 300, 150))) {
                            // Capture, crop, and display the image
                            captureAndDisplay(detectedObject.boundingBox, image)
                            break
                        }
                    }
                }
                imageProxy.close()
            }
            .addOnFailureListener { e ->
                Log.e("ishaq Object Detection", "Error detecting objects", e)
                imageProxy.close()
            }
    }

    private fun captureAndDisplay(boundingBox: Rect, image: InputImage) {
        val bitmap = image.bitmapInternal ?: return

        // Crop the detected ID card
        val croppedBitmap = Bitmap.createBitmap(bitmap, boundingBox.left, boundingBox.top, boundingBox.width(), boundingBox.height())

        // Display the cropped image
        runOnUiThread {
            findViewById<ImageView>(R.id.croppedImageView).setImageBitmap(croppedBitmap)
        }
    }
}