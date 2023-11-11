package com.example.deepbeyond

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.Core
import org.opencv.imgproc.Imgproc

// 輪郭抽出関数
fun getContours(srcMat: Mat): List<MatOfPoint> {
    // グレースケール化
    val grayMat = Mat()
    Imgproc.cvtColor(srcMat, grayMat, Imgproc.COLOR_BGRA2GRAY)

    // ノイズ除去
    Imgproc.medianBlur(grayMat, grayMat, 7)

    // 輪郭抽出
    val hierarchy: Mat = Mat.zeros(Size(5.0, 5.0), CvType.CV_8UC1)
    val contours: List<MatOfPoint> = ArrayList()
    Imgproc.findContours(
        grayMat,
        contours,
        hierarchy,
        Imgproc.RETR_EXTERNAL,
        Imgproc.CHAIN_APPROX_TC89_L1
    )
    grayMat.release()

    return contours
}

// 輪郭情報から近似輪郭の座標を得る
fun getContourVertex(contours: List<MatOfPoint>, srcMat: Mat):
        Pair<MutableList<Pair<Double, Double>>, Rect> {
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

    // 例外処理
    if (contours.size == 0){
        throw IllegalArgumentException("contours size is 0")
    }

    // MatOfPointからMatOfPoint2fに変換
    val contour = contours[0]
    val contour2f = MatOfPoint2f(*contour.toArray())  // contours.size = 1

    // 輪郭線の長さ
    val arclen = Imgproc.arcLength(contour2f, true)

    // 外接矩形計算
    val boundingRect = Imgproc.boundingRect(contour2f)

    // 外接矩形が画像サイズに対して比較的小さい場合はノイズとして無視(if判定でFalseとする)
    val approx = MatOfPoint2f()
    if (h * scale < boundingRect.height && w * scale < boundingRect.width) {
        // 輪郭線の近似
        val epsilon = 0.005 * arclen

        Imgproc.approxPolyDP(contour2f, approx, epsilon, true)

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

// 輪郭とキ甲の交点をアルファ値によって判断する
fun getIntersection(witherPosX: Int, bboxY: Int, bboxH: Int, srcMat: Mat): Array<Array<Int>> {
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
    val highY = bboxY + bboxH - reversedAlpha.indexOf(255) + 1      // 455, 175    455, 645

    return arrayOf(arrayOf(witherPosX, lowY), arrayOf(witherPosX, highY))
}

// キ甲の座標情報を取得
fun getWithersPosition(contourVertex:MutableList<Pair<Double, Double>>,
                       bboxPosition:Rect, srcMat: Mat)
: Triple<Int, Array<Array<Int>>, Int> {

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
    val lastToesPosX = toesPosXs.max().toInt()

    //
    // 3. キ甲の長さを探索(キ甲と輪郭の交点を探索)
    //

    // 輪郭とキ甲の直線の交点(2つ)を探索
    val witherPos = getIntersection(witherPosX, bboxY, bboxH, srcMat)

    //
    // 4. キ甲と輪郭の交点を修正
    //

    // 足先に線分の頂点が位置していない場合、足先の座標に変更
    // 2点の中でY座標が大きい方の頂点がtoesYよりも小さければ変更
    val toesPosY = toesPosYs.average().toInt()

    if (witherPos[0][1] > witherPos[1][1]) {
        if (witherPos[0][1] < toesPosY) {
            witherPos[0][1] = toesPosY
        }
    } else {
        if (witherPos[1][1] < toesPosY) {
            witherPos[1][1] = toesPosY
        }
    }

    return Triple(witherPosX, witherPos, lastToesPosX)
}

// 胴を探索
fun getTorso(contourVertex:MutableList<Pair<Double, Double>>,
             bboxPosition:Rect, witherPosX: Int): Int {
    val bboxY = bboxPosition.y
    val bboxH = bboxPosition.height

    //
    //1.探索範囲を設定
    //

    //胴の終点を探索するため外接矩形の「キ甲より右側（尻側）」「上側1/3」の範囲を見る
    val onethirdH = bboxH / 3 + bboxY       // 279

    //
    //2.胴のx座標を探索
    //

    //時系列t-1における傾きとx座標の情報
    var prevTilt = 0.0
    var prevX = 0

    //胴のx座標の頂点
    var torsoPosX = 0

    //胴の終点であるフラグ
    var torsoFlg = false

    for(i in contourVertex.indices){
        val (x1, y1) = contourVertex[i]  // 始点

        // 配列外参照にならないようにループさせる
        val (x2, y2) = if (i == contourVertex.size - 1) {
            contourVertex[0]  // 終点
        } else {
            contourVertex[i + 1]  // 終点
        }

        //「キ甲より右側（尻側）」「上側1/3」の範囲以外ならば処理しない
        if (x1 < witherPosX || y1 > onethirdH) {
            continue
        }

        var distanceX = x2 - x1
        val distanceY = y2 - y1

        // 0除算対策
        if (distanceX == 0.0){
            distanceX += 0.00001
        }

        val tilt: Double = Math.round(distanceY / distanceX * 10.0) / 10.0  // 小数点第1位
        Log.d("HorseInfo", "傾き：$tilt")

        //傾きが正->負->正になった場合、負の箇所を胴の終点とする
        if (torsoFlg && tilt > 0.0) {
            torsoPosX = prevX
            break
        }

        if (tilt <= 0.0 && prevTilt != 0.0 ){
            torsoFlg = true
        }

        prevTilt = tilt
        prevX = x1.toInt()
    }

    return torsoPosX
}

// 首を探索
fun getNeck(contourVertex:MutableList<Pair<Double, Double>>, witherPos: Array<Array<Int>>): Double {
    // yの値が最も小さい頂点を始点とする
    val sortedContour = contourVertex.sortedBy { it.second }
    val start = sortedContour[0]
    val end = Pair(witherPos[0][0].toDouble(), witherPos[0][1].toDouble())

    val deltaX = Math.abs(end.first - start.first)
    val deltaY = Math.abs(end.second - start.second)
    val neckLength = Math.sqrt((deltaX * deltaX + deltaY * deltaY))

    return (Math.round(neckLength * 10.0)).toDouble() / 10.0    // 小数点第1位
}

// 尻のx座標を探索
fun getHip(torsoPosX: Int, bboxPosition: Rect, img: Mat): Int {
    //
    // 1. 探索範囲を設定
    //
    val bboxY = bboxPosition.y
    val bboxH = bboxPosition.height
    val bboxW = bboxPosition.width

    // 画像の尻部分のみ着目
    val baseImg = Mat(img, Rect(torsoPosX, bboxY, bboxW, bboxH / 2))
    val h = baseImg.rows()
    val w = baseImg.cols()

    //
    // 2. 尻の先端のx座標を探索
    //
    val hipPosLog = mutableListOf<Int>()

    // 画像の大きさによってノイズ除去の度合を変更
    val ksize: Int
    if (img.cols() < 150) {
        ksize = 3
    } else {
        ksize = 5
    }

    //
    //  以下, 非修正
    //

    for (itr in 1..3) {
        for (alpha in listOf(1.5, 2.5, 4.5)) {
            val adjustedImg = Mat()
            Core.convertScaleAbs(baseImg, adjustedImg, alpha)
            Imgproc.cvtColor(adjustedImg, adjustedImg, Imgproc.COLOR_BGR2GRAY)

            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(adjustedImg, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            // 二値化
            val adaptiveThresholdImg = Mat()
            Imgproc.adaptiveThreshold(adjustedImg, adaptiveThresholdImg, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 13, 20.0)

            // lsdのための画像作成
            val lsdImg = Mat.zeros(adaptiveThresholdImg.size(), CvType.CV_8UC1)

            val tmp = Mat()

            Imgproc.createLineSegmentDetector().detect(adaptiveThresholdImg, tmp)


//            for (line in Imgproc.createLineSegmentDetector().detect(adaptiveThresholdImg)) {
//                val x1 = line[0].toInt()
//                val y1 = line[1].toInt()
//                val x2 = line[2].toInt()
//                val y2 = line[3].toInt()
//                Imgproc.line(lsdImg, Point(x1.toDouble(), y1.toDouble()), Point(x2.toDouble(), y2.toDouble()), Scalar(255.0), 1)
//            }


            Core.bitwise_not(lsdImg, lsdImg)

            Core.bitwise_not(lsdImg, lsdImg)
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
            Imgproc.dilate(lsdImg, lsdImg, kernel, Point(-1.0, -1.0), itr)
            Core.bitwise_not(lsdImg, lsdImg)

            var bottomRightXPos = 0
            for (x in w - 1 downTo 0) {
                if (lsdImg[h - 1, x][0] <= 0.0) {
                    bottomRightXPos = x
                    break
                }
            }

            val contour = MatOfPoint()
            var drawFlag = false
            for (point in contours[0].toList()) {
                val x = point.x.toInt()
                val y = point.y.toInt()
                if (bottomRightXPos - x < 10) {
                    drawFlag = true
                }
                if (drawFlag) {
                    contour.push_back(MatOfPoint(Point(x.toDouble(), y.toDouble())))
                }
            }

            for (i in 0 until contour.size() - 1) {
                val x1 = contours[i].get(0,0)[0]
                val y1 = contours[i].get(0,0)[1]
                val x2 = contours[i + 1].get(0,0)[0]
                val y2 = contours[i + 1].get(0,0)[1]

            }

            for (j in 0 until 4) {
                Imgproc.medianBlur(adjustedImg, adjustedImg, ksize)
            }

            val candPosXS = mutableListOf<Int>()
            var prevX: Int? = null
            var endFlag = false
            var length = 0

            for (y in h - 1 downTo h / 2) {
                if (endFlag) {
                    break
                }
                for (x in 0 until w - 10) {
                    if (adjustedImg[y, x][0] <= 0.0) {
                        if (prevX != null) {
                            if (0 <= prevX - x && prevX - x < 5) {
                                length++
                            } else if (prevX - x > 10) {
                                continue
                            } else {
                                length = 0
                            }
                        }
                        if (length > 10) {
                            candPosXS.add(x)
                        }
                        prevX = x
                        if (length > 10) {
                            endFlag = true
                            break
                        }
                    }
                }
            }

            if (candPosXS.isNotEmpty()) {
                hipPosLog.add(candPosXS.maxOrNull()!!)
            }
        }
        if (hipPosLog.isNotEmpty()) {
            break
        }
    }

    var hipPosX = hipPosLog.maxOrNull() ?: 0
    hipPosX += torsoPosX

    return hipPosX
}

// ともを探索
fun getHindlimb(torsoPosX: Int, bboxPosition:Rect,
                contourVertex:MutableList<Pair<Double, Double>>, lastTpesPosX: Int, img: Mat): Int {
    // 尻のx座標を探索
    val hipPosX = getHip(torsoPosX, bboxPosition, img)

    // ともの始点を探索
    var limitCnt = 0
    var hindlimbPosX = 0
    var hindlimbPosY = Double.POSITIVE_INFINITY

    for (i in 0 until contourVertex.size - 1) {
        val (x1, y1) = contourVertex[i]  // 始点

        // 始点が外接矩形の左半分ならばスキップ
        if (x1 < lastTpesPosX) {
            continue
        }

        // 条件を満たす7頂点を見れば処理終了
        if (limitCnt > 7) {
            break
        }

        limitCnt++
        // 条件を満たす7点の中でy座標が最小の点がともの始点
        if (y1 < hindlimbPosY) {
            hindlimbPosY = y1
            hindlimbPosX = x1.toInt()
        }
    }

    return hipPosX - hindlimbPosX
}