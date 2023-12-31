package com.example.deepbeyond

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import com.example.deepbeyond.ui.theme.DeepBeyondTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Size
import org.opencv.dnn.SegmentationModel
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel


const val DISPLAY_SIZE_DP = 300 // ディスプレイサイズ(dp)

class MainActivity : ComponentActivity() {
    private var canvasPixel = 0f    // リサイズ用
    private var resizedW = 0
    private var resizedH = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        // OpenCV初期化
        if (!OpenCVLoader.initDebug()) {
            println("error")
        }
        super.onCreate(savedInstanceState)

        val context = applicationContext
        val metrics = context.resources.displayMetrics
        canvasPixel = DISPLAY_SIZE_DP * metrics.density     // dp -> pixel

        setContent {
            DeepBeyondTheme {
                Surface(
                    color = MaterialTheme.colorScheme.background
                ) {
                    SelectedButton(context)
                }
            }
        }
    }

    override fun onDestroy() {
        // アプリが閉じる際にInterpreterをクローズする
        interpreter.close()
        super.onDestroy()
    }

    // Assetsフォルダ内にあるモデル、ラベルの名称
    companion object {
        private const val MODEL_FILE_NAME = "lite-model_mobilenetv2-dm05-coco_dr_1.tflite"
    }

    // TF Liteモデルを扱うためのラッパーを含んだinterpreter
    private val interpreter: Interpreter by lazy {
        Interpreter(loadModel())
    }

    // AssetsフォルダからTF Liteモデルを読み込む
    private fun loadModel(fileName: String = MODEL_FILE_NAME): ByteBuffer {
        lateinit var modelBuffer: ByteBuffer
        var file: AssetFileDescriptor? = null
        try {
            file = assets.openFd(fileName)
            val inputStream = FileInputStream(file.fileDescriptor)
            val fileChannel = inputStream.channel
            modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, file.startOffset, file.declaredLength)
        } catch (e: Exception) {
            Toast.makeText(this, "モデルファイル読み込みエラー", Toast.LENGTH_SHORT).show()
            finish()
        } finally {
            file?.close()
        }
        return modelBuffer
    }

    // Composableでない関数の場合, contextを渡す必要
    private fun uri2bitmap (uri: Uri, context: Context): Bitmap {
        val source = ImageDecoder.createSource(context.contentResolver, uri)
        return ImageDecoder.decodeBitmap(source){ decoder, _, _ ->
            decoder.isMutableRequired = true
        }
    }

    // キャンバスサイズの幅に合わせてリサイズ
    private fun resizedBitmap(bitmap: Bitmap): Bitmap{
        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()
        val ratio = canvasPixel / w
        resizedW = (w * ratio).toInt()
        resizedH = (h * ratio).toInt()
        return Bitmap.createScaledBitmap(bitmap, resizedW, resizedH, true)
    }

    // キャンバスの中心に描画できるように位置を計算
    private fun calcCanvasCenter(canvasH: Float): Offset{
        val top = (canvasH - resizedH) / 2
        return Offset(x = 0f, y = top)
    }

    private fun segment(bitmap: Bitmap) : Mat{
        val seg = Segmantation(interpreter)
        return seg.segment(bitmap)
    }

    private fun mainProcess(bitmap: Bitmap): Array<Double> {
        // 入力画像: マスク部分だけ切り出した画像
        val srcMat = segment(bitmap)

        // デバッグ：画像確認用
         // conformImage(srcMat)

        // 輪郭
        val contours = getContours(srcMat)

        // 特定の条件を満たす輪郭の座標を取得
        val (contourVertex, bboxPosition) = getContourVertex(contours, srcMat)

        // キ甲を探索
        val (witherPosX, witherPos, lastToesPosX) = getWithersPosition(contourVertex, bboxPosition, srcMat)
        val witherLength = witherPos[1][1] - witherPos[0][1].toDouble()
        Log.d("HorseInfo", "キ甲の長さ：$witherLength")

        // 胴を探索
        val torsoPosX = getTorso(contourVertex, bboxPosition, witherPosX)
        val torsoLength = torsoPosX - witherPos[0][0].toDouble()
        Log.d("HorseInfo", "胴の長さ: $torsoLength")

        // 首
        val neckLength = getNeck(contourVertex, witherPos)
        Log.d("HorseInfo", "首の長さ: $neckLength")

        // 繋   不要かもしれない

        // とも
        val hindLimbLength = getHindlimb(torsoPosX, bboxPosition, contourVertex, lastToesPosX, srcMat).toDouble()
        Log.d("HorseInfo", "ともの長さ: $hindLimbLength")

        srcMat.release()

        return arrayOf(witherLength, torsoLength, neckLength, hindLimbLength)
    }

    @Composable
    private fun SelectedButton(context: Context) {
        // 選択された画像のURIを保持
        var imageUri by remember { mutableStateOf<Uri?>(null) }

        // 馬情報
        var witherLength by remember { mutableStateOf(0.0) }        // キ甲の長さ: 623
        var torsoLength by remember { mutableStateOf(0.0) }         // 胴の長さ: 351
        var neckLength by remember { mutableStateOf(0.0) }          // 首の長さ: 303.7
        var hindLimbLength by remember {mutableStateOf(0.0)}        // ともの長さ: 178

        // ギャラリーから画像を選択するためのアクティビティ結果コントラクトを宣言
        val launcher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            imageUri = uri
        }

        // 非同期処理の結果をUIに反映する
        LaunchedEffect(imageUri) {
            if (imageUri == null) {
                witherLength = 0.0
                torsoLength = 0.0
                neckLength = 0.0
                hindLimbLength = 0.0
            }
            imageUri?.let { uri ->
                // uriをBitmapに変換
                var bitmap = uri2bitmap(uri, context)
                // リサイズ
                bitmap = resizedBitmap(bitmap)

                // 推論を実行
                val output = withContext(Dispatchers.Default) {
                    mainProcess(bitmap)
                }

                // 情報を更新
                withContext(Dispatchers.Main) {
                    witherLength = output[0]
                    torsoLength = output[1]
                    neckLength = output[2]
                    hindLimbLength = output[3]
                }
            }
        }

        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ){
            Canvas(modifier = Modifier.size(DISPLAY_SIZE_DP.dp), onDraw = {
                if (imageUri == null) {drawRect(color = Color.Gray)}
                imageUri?.let { uri ->
                    // RGBA画像が表示されているか確認用
                    // drawRect(color = Color.Red)

                    // uriをBitmapに変換
                    var bitmap = uri2bitmap(uri, context)
                    // リサイズ
                    bitmap = resizedBitmap(bitmap)

                    val imageBitmap = bitmap.asImageBitmap()
                    drawImage(imageBitmap, topLeft = calcCanvasCenter(size.height))
                }
            })
            Button(
                onClick = {
                    // 画像ギャラリーの表示 (選択画像URIはimageUriに格納)
                    launcher.launch("image/*")
                },
                modifier = Modifier.padding(16.dp)
            ){
                Text(text = "SELECT IMAGE")
            }
            Text("キ甲： $witherLength")
            Text("胴： $torsoLength")
            Text("首： $neckLength")
            Text("とも： $hindLimbLength")
        }
    }
}