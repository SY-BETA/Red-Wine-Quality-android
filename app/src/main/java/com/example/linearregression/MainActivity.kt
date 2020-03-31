package com.example.linearregression

import android.content.Context
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import kotlinx.android.synthetic.main.activity_main.*
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        //ワインのテスト用データ
        val inputArray = floatArrayOf(7.1f, 0.46f, 0.2f, 1.9f, 0.077f, 28f, 54f, 0.9956f, 3.37f, 0.64f, 10.4f)
        //テンソルの生成:　引数(floatArray, テンソルのサイズ)
        val inputTensor = Tensor.fromBlob(inputArray, longArrayOf(1,11))
        //モデルのロード
        val module = Module.load(assetFilePath(this, "wineModel.pt"))
        //推論
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray
        //結果の表示
        result.text ="予測値: ${scores[0]}"
        label.text = "正解ラベル：6"
    }

    //assetフォルダからパスを取得する関数
    fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
            return file.absolutePath
        }
    }
}
