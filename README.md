# carbonstock-backend
## How to use:

1. Download model di drive
2. Masukkin model ke folder model
3. python -m venv myenv (jika belum bikin env, jalankan sekali saja)
4. myenv\\Scripts\\activate   # Windows
5. pip install tensorflow keras numpy flask pillow
6. py asd.py 

## Dokumentasi api

1. http://127.0.0.1:5000/classify:

    - untuk ngeluarin class dari gambar
    - input: 
        > key: file, value: image (.jpg, .png, jpeg)
    - output:
        ```
        {
            "class": "Rendah",
            "confidence": 0.6493560671806335
        }
        ```

2. http://127.0.0.1:5000/predict_xxx?size=yyy&type=zzz

    - untuk ngeluarin estimasi karbon
    - input: 
        > key: file, value: image (.jpg, .png, jpeg)
    - output:
        ```
        {
               "predicted_value": "18.43 ton/ha"
        }
        ```
    - params:
        - predict_xxx : xxx = pilih model ada 3 macem: vgg, cnn, resnet
        - size=yyy : yyy = ukuran plot ada 5 macem: 20x20, 10x10, 5x5, 1x1, gabung
        - type=zzz : zzz = tipe gambar dari mana, ada 3 macem: drone, gee, mix
