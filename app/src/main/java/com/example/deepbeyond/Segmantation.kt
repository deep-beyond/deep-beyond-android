package com.example.deepbeyond

// 参考
// https://github.com/joaquim-verges/DeeplabAndroid/blob/master/app/src/main/java/com/joaquimverges/deeplabandroid/ImageSegmentationAnalyzer.kt


import android.graphics.Bitmap
import android.graphics.Color
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.ByteBuffer
import java.nio.ByteOrder


//1 to "Plane",
//2 to "Bycicle",
//3 to "Bird",
//4 to "Boat",
//5 to "Bottle",
//6 to "Bus",
//7 to "Car",
//8 to "Cat",
//9 to "Chair",
//10 to "Cow",
//11 to "Table",
//12 to "Dog",
//13 to "Horse",
//14 to "Bike",
//15 to "Person",
//16 to "Plant",
//17 to "Sheep",
//18 to "Sofa",
//19 to "Train",
//20 to "TV"

class Segmantation(
    private val interpreter: Interpreter
){
    companion object {
        // モデルのinputとoutputサイズ
        private const val INPUT_SIZE = 513
        private const val OUTPUT_SIZE = 513

        // クラス数
        private const val NUM_CLASS = 21

        private const val NORMALIZE_MEAN = 0f
        private const val NORMALIZE_STD = 128f
    }

    private val tfImageBuffer = TensorImage(DataType.FLOAT32)

    private val tfImageProcessor by lazy {
        ImageProcessor.Builder()
            .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR)) // 画像サイズの変更
            .add(NormalizeOp(NORMALIZE_MEAN, NORMALIZE_STD)) //　正規化
            .build()
    }

    fun segment(targetBitmap: Bitmap): Mat{
        // Bitmap -> TensorFlowBuffer
        tfImageBuffer.load(targetBitmap)

        // 前処理(画像サイズの変更、正規化)
        val tensorImage = tfImageProcessor.process(tfImageBuffer)

        // 入出力用のバッファを作成
        val inputBuf = tensorImage.buffer
       val outputBuf= ByteBuffer.allocateDirect(1 * OUTPUT_SIZE * OUTPUT_SIZE * NUM_CLASS * 4)
        outputBuf.order(ByteOrder.nativeOrder())

        // 推論実行
        interpreter.run(inputBuf, outputBuf)
        // interpreter.close()

        // 描画
        val maskBitmap = draw(outputBuf, targetBitmap.width, targetBitmap.height)

        // 入力画像を切り抜き        // 入力画像にマスク部分だけ切り出し
        val rawMat = Mat()
        val maskMat= Mat()
        val srcMat = Mat()
        Utils.bitmapToMat(targetBitmap, rawMat)
        Utils.bitmapToMat(maskBitmap, maskMat)
        rawMat.copyTo(srcMat, maskMat)

        rawMat.release()
        maskMat.release()

        return srcMat
    }

    fun draw(resultBuf: ByteBuffer, w: Int, h: Int) : Bitmap {
        val maskBitmap = Bitmap.createBitmap(OUTPUT_SIZE, OUTPUT_SIZE, Bitmap.Config.ARGB_8888)
        var maxScore = 0f
        var score: Float
        val seenLabels = mutableMapOf<Int, Int>()
        val segmentBits = Array(OUTPUT_SIZE) { IntArray(OUTPUT_SIZE) }
        val segmentColors = IntArray(NUM_CLASS)

        // 対象物と色を紐づける (馬以外は透明, 馬は白色 <- 黒色にするとグレースケール化の時に全て黒になるのを防ぐため, 白色)
        for (i in 0 until NUM_CLASS) {
            when (i) {
                13 -> segmentColors[i] = Color.WHITE
                else -> segmentColors[i] = Color.TRANSPARENT
            }
        }

        for (y in 0 until OUTPUT_SIZE) {
            for (x in 0 until OUTPUT_SIZE) {
                segmentBits[x][y] = 0
                // find the highest output value from all the labels
                for (c in 0 until NUM_CLASS) {
                    score = resultBuf.getFloat((y * OUTPUT_SIZE * NUM_CLASS + x * NUM_CLASS + c) * 4)
                    if (c == 0 || score > maxScore) {
                        maxScore = score
                        segmentBits[x][y] = c
                    }
                }
                // keep track of all seen labels, counting how many pixels they cover
                val labelIndex = segmentBits[x][y]
                if (labelIndex != 0) {
                    val labelPixelCount = seenLabels[labelIndex] ?: 0
                    seenLabels[labelIndex] = labelPixelCount.inc()
                }
                // finally, get the color value for that label and set it in the mask bitmap
                val pixelColor = segmentColors[labelIndex]
                maskBitmap.setPixel(x, y, pixelColor)
            }
        }

        return Bitmap.createScaledBitmap(maskBitmap, w, h, false)
    }

}