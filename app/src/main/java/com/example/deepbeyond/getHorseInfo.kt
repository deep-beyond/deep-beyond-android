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
    var conditionBboxW = 0  // 条件を満たす外接矩形の幅

    // MatOfPointからMatOfPoint2fに変換
    val contour = MatOfPoint2f(*contours[0].toArray())  // contours.size = 1

    // 輪郭線の長さ
    val arclen = Imgproc.arcLength(contour, true)

    // 外接矩形計算
    val boundingRect = Imgproc.boundingRect(contour)

    // 外接矩形が画像サイズに対して比較的小さい場合はノイズとして無視(if判定でFalseとする)
    val approx = MatOfPoint2f()
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

    // 近似輪郭の頂点を格納
    for (v in approx.toList()) {
        val pos = Pair(v.x, v.y)
        contourVertex.add(pos)
    }

    // 画像内の馬に対する外接矩形の座標情報(4点)
    val bboxPosition = Rect(conditionBboxX, conditionBboxY, conditionBboxW, conditionBboxH)
    return Pair(contourVertex, bboxPosition)
}

fun getWithersPosition(test:  Pair<MutableList<Pair<Double, Double>>, Rect>
): Triple<Int, List<Pair<Double, Double>>, Double?> {
    val bboxX = test.second.x
    val bboxY = test.second.y
    val bboxH = test.second.height
    val bboxW = test.second.width

    // Define constants
    val onethirdH = bboxH / 3 + bboxY
    val quarterW = bboxW / 4 + bboxX
    val lowerLine = (bboxY + bboxH - bboxH * 0.1).toInt()

    // Initialize variables
    var prevDistanceY = 0.0
    val toesPosXs = mutableListOf<Double>()
    val toesPosYs = mutableListOf<Double>()

    for (i in 0 until test.first.size - 1) {
        val (x1, y1) = test.first[i]
        val (x2, y2) = test.first[i + 1]

        if (y1 < onethirdH && y2 < onethirdH || x1 < quarterW) {
            continue
        }

        val distanceX = x2 - x1
        val distanceY = y2 - y1

        // Prevent division by zero
        val tilt = if (distanceX != 0.0) distanceY.toDouble() / distanceX.toDouble() else 0.0

        if (prevDistanceY < -bboxH * 2 / 7 && Math.abs(tilt) < 3) {
            break
        }

        if (y1 > lowerLine) {
            toesPosXs.add(x1)
            toesPosYs.add(y1)

        }

        prevDistanceY = distanceY
    }

    if (toesPosXs.isEmpty()) {
        throw IllegalArgumentException("Can't look for toes vertex")
    }

    val witherPosX = toesPosXs.average().toInt()
    val lastToesPosX = toesPosXs.maxOrNull()


    // More code for the intersection and other parts can be added here.

    return Triple(witherPosX, emptyList(), lastToesPosX)
}