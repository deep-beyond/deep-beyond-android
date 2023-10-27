package com.example.deepbeyond

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

// 輪郭抽出関数
fun getContours(srcMat: Mat): List<MatOfPoint> {

    // グレースケール化
    Imgproc.cvtColor(srcMat, srcMat, Imgproc.COLOR_BGR2GRAY)

    // 輪郭抽出
    val hierarchy: Mat = Mat.zeros(Size(5.0, 5.0), CvType.CV_8UC1)
    val contours: List<MatOfPoint> = ArrayList()

    Imgproc.findContours(
        srcMat,
        contours,
        hierarchy,
        Imgproc.RETR_EXTERNAL,
        Imgproc.CHAIN_APPROX_TC89_L1
    )
    return contours
}

// 輪郭情報から近似輪郭の座標を得る
fun getContourVertex(contours: List<MatOfPoint>, srcMat: Mat): Pair<MutableList<Pair<Double, Double>>, Rect> {
    // 輪郭の座標リスト
    val contourVertex = mutableListOf<Pair<Double, Double>>()

    // 画像の高さ, 幅
    val h = srcMat.rows()
    val w = srcMat.cols()

    val scale = 0.6         // ノイズ除去のためのスケール比
    var conditionBboxX = 0  // 条件を満たす外接矩形のx座標
    var conditionBboxY = 0  // 条件を満たす外接矩形のy座標
    var conditionBboxH = 0  // 条件を満たす外接矩形の高さ
    var conditionBboxW = 0  //

    val approx = MatOfPoint2f()
    for (contour in contours) {
        // MatOfPointからMatOfPoint2fに変換
        val contour = MatOfPoint2f(*contour.toArray())

        // 輪郭線の長さ
        val arclen = Imgproc.arcLength(contour, true)

        // 外接矩形計算
        val boundingRect = Imgproc.boundingRect(contour)

        // 外接矩形が画像サイズに対して比較的小さい場合はノイズとして無視(if判定でFalseとする)
        if (h * scale < boundingRect.height && w * scale < boundingRect.width) {
            // 輪郭線の近似
            val epsilon = 0.005 * arclen

            Imgproc.approxPolyDP(contour, approx, epsilon, true)

            // 条件を満たす外接矩形の情報を記録
            conditionBboxX = boundingRect.x
            conditionBboxY = boundingRect.y
            conditionBboxH = boundingRect.height
            conditionBboxW = boundingRect.width
        }
    }

    for (v in approx.toList()) {
        val pos = Pair(v.x, v.y)
        contourVertex.add(pos)
    }
    val bboxPosition = Rect(conditionBboxX, conditionBboxY, conditionBboxW, conditionBboxH)
    return Pair(contourVertex, bboxPosition)
}
