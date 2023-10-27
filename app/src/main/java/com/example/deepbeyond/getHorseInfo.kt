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

fun getIntersection(witherPosX: Int, bboxY: Int, bboxH: Int, srcMat: Mat): List<List<Int>> {
    // アルファ値を取得
    val alpha = mutableListOf<Int>()   // アルファ値を格納するリスト
    for (y in bboxY until bboxY+bboxH) {
        val alphaValue = srcMat.get(y, witherPosX)[3].toInt()
        alpha.add(alphaValue)
    }

    // 画像上部における交点
    val lowY = bboxY + alpha.indexOf(255)

    // 画像下部における交点
    val reversedAlpha = alpha.reversed()
    val highY = bboxY + bboxH - reversedAlpha.indexOf(255) + 1

    return listOf(listOf(witherPosX, lowY), listOf(witherPosX, highY))
}

fun getWithersPosition(vertexAndBbox:  Pair<MutableList<Pair<Double, Double>>, Rect>, srcMat: Mat
): Triple<Int, List<Pair<Double, Double>>, Double?> {
    val contourVertex = vertexAndBbox.first
    val bboxPosition = vertexAndBbox.second

    val bboxX = bboxPosition.x
    val bboxY = bboxPosition.y
    val bboxH = bboxPosition.height
    val bboxW = bboxPosition.width

    //
    // 1. 探索範囲を設定
    //

    // 前足を探索するため外接矩形の「上部1/3～上部3/3」「左1/4~左4/4」の範囲を見る
    val onethirdH = bboxH / 3 + bboxY
    val quarterW = bboxW / 4 + bboxX

    // 足先を判定するためのライン（外接矩形の下から0.1のライン）
    val lowerLine = (bboxY + bboxH - bboxH * 0.1).toInt()

    //
    // 2. キ甲のx座標を探索
    //

    // 時刻t-1における辺分の情報
    var prevDistanceY = 0.0

    // 足先のX座標の頂点
    val toesPosXs = mutableListOf<Double>()
    val toesPosYs = mutableListOf<Double>()

    // -1 にしないとfor文最後の実行において終点で配列外参照になる
    for (i in 0 until contourVertex.size - 1) {
        val (x1, y1) = contourVertex[i]     // 始点
        val (x2, y2) = contourVertex[i + 1] // 終点

        // 辺分が上部1/3の位置に属しているなら処理しない
        if (y1 < onethirdH && y2 < onethirdH || x1 < quarterW) {
            continue
        }

        val distanceX = x2 - x1
        val distanceY = y2 - y1

        // 0除算対策
        val tilt = if (distanceX != 0.0) distanceY / distanceX else 0.0

        // 時刻t-1に線分が急激に右肩上がりになり、時刻tで傾きが弱まった場合
        if (prevDistanceY < -bboxH * 2 / 7 && Math.abs(tilt) < 3) {
            break
        }

        // 足先の頂点を探索
        if (y1 > lowerLine) {
            toesPosXs.add(x1)
            toesPosYs.add(y1)
        }

        // 時刻tの情報を時刻t-1の情報にする
        prevDistanceY = distanceY
    }

    if (toesPosXs.isEmpty()) {
        throw IllegalArgumentException("Can't look for toes vertex")
    }

    // キ甲のX座標：前足の頂点らの中点
    val witherPosX = toesPosXs.average().toInt()

    // 前足の頂点らの最も右端の点
    val lastToesPosX = toesPosXs.maxOrNull()

    //
    // 3. キ甲の長さを探索(キ甲と輪郭の交点を探索
    //

    // アルファ値を取得
    val alpha = mutableListOf<Int>()   // アルファ値を格納するリスト
    for (y in bboxY until bboxY+bboxH) {
        val alphaValue = srcMat.get(y, witherPosX)[3].toInt()
        alpha.add(alphaValue)
    }

    // 画像上部における交点
    val lowY = bboxY + alpha.indexOf(255)

    // 画像下部における交点
    val reversedAlpha = alpha.reversed()
    val highY = bboxY + bboxH - reversedAlpha.indexOf(255) + 1

    val witherPos = listOf(listOf(witherPosX, lowY), listOf(witherPosX, highY))
//    val witherPos = getIntersection(witherPosX, bboxY, bboxH, srcMat)

    //
    // 4. キ甲と輪郭の交点を修正
    //

    // 足先に線分の頂点が位置していない場合、足先の座標に変更
    // 2点の中でY座標が大きい方の頂点がtoesYよりも小さければ変更
//    val toesPosY = toesPosYs.sum() / toesPosYs.size

//    if (witherPos[0][1] > witherPos[1][1]) {
//        if (witherPos[0][1] < toesPosY) {
//            witherPos[0][1] = toesPosY
//        }
//    } else {
//        if (witherPos[1][1] < toesPosY) {
//            witherPos[1][1] = toesPosY
//        }
//    }

    return Triple(witherPosX, emptyList(), lastToesPosX)
}

fun getTorso(contour: ,bboxPotion:, witherPosX:, descimg:, args:): {
    val bboxY = bboxPotion[1]
    val bboxH = bboxPotion[2]

    //
    //1.探索範囲を設定
    //

    //胴の終点を探索するため外接矩形の「キ甲より右側（尻側）」「上側1/3」の範囲を見る
    val onethirdH = bboxH / 3 + bboxY

    //
    //2.胴のx座標を探索
    //

    //時系列t-1における傾きとx座標の情報
    var prevTilt = 0
    var prevX = 0

    //胴のx座標の頂点
    var torsoPosX = 0

    //胴の終点であるフラグ
    var torsoFIg = false

    for(i in contour.indices){
        val (x1, y1) = contour[i]  // 始点

        // 配列外参照にならないようにループさせる
        val (x2, y2) = if (i == contour.size - 1) {
            contour[0]  // 終点
        } else {
            contour[i + 1]  // 終点
        }

        //「キ甲より右側（尻側）」「上側1/3」の範囲以外ならば処理しない
        if (x1 < witherPosX || y1 > onethirdH) {
            continue
        }

        val distanceX = x2 - x1
        val distanceY = y2 - y1
        val tilt = (distanceY.toDouble() / distanceX.toDouble()).round(1)

        //傾きが正->負->正になった場合、負の箇所を胴の終点とする
        if (torsoFIg == true && tilt<0) {
            torsoPosX = prevX
            break
        }

        if (tilt <= 0 && prevTilt > 0 ){
            torsoFlg = true
        }

        prevTilt = tilt
        prevX = x1
    }

    return torsePosX
}