import socket
import numpy
import cv2
import numpy as np
import struct
import math
from datetime import datetime

controll_raspi_port = 50007  # 制御ラズパイのポート
controll_raspi_ipaddr = '192.168.179.5'  # 制御ラズパイのIPアドレス
camera_raspi_ipaddr = controll_raspi_ipaddr  # カメラのラズパイアドレス　制御と同じにしている
camera1_raspi_port = 5569  # カメラ1（正面カメラ）のラズパイポート番号
camera2_raspi_port = 5570  # カメラ2（下カメラ）のラズパイポート番号

min_detect_area = 1e3  # 最小検出面積
camera2_balloon_detection_threshold = 5e3  # 下カメラで風船を検出したとき、風船割るモードに遷移する面積（ピクセル数）のしきい値
Balloon_non_detection_waiting_time = 2 * 10  # 風船が見つからないときに待機するフレーム数。2秒x10fps

camera1_resolution = [640, 480]  # カメラ1の解像度 (width, hight)
camera2_resolution = [640, 480]  # カメラ2の解像度 (width, hight)

max_motor_val = 50  # モータの最大値　0-100で入力する。出力を抑えたいときに使う。
motor_power_const = 1  # モータ出力を一定割合で抑える。0-1の間にする。

debug_mode = False  # マスク画像、2値画像などが表示される

# 風船の色範囲を設定する。正面カメラと下カメラで異なる値を設定できる。現在の設定値はJAMSTEC多目的プールで赤色風船を検出する設定
mask_hsv_cam1 = [110,180,0, 255,0,255] # H_下限  H_上限 S_下限 S_上限 V_下限 V_上限
mask_hsv_cam2 = [120,180,0, 255,0,255] # H_下限  H_上限 S_下限 S_上限 V_下限 V_上限


def getimage(HOST, PORT):
    # ここを参考にしています。
    # https://qiita.com/sourcekatu/items/ddc0f372c775e6bf53f1

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    sock.send(b'test')

    # バイト型
    buf = b''
    recvlen = 100
    while recvlen > 0:
        receivedstr = sock.recv(1024 * 8)
        recvlen = len(receivedstr)
        buf += receivedstr
    sock.close()
    narray = numpy.fromstring(buf, dtype='uint8')
    return cv2.imdecode(narray, 1)


def detect_contour(src, mask_hsv ,name):
    # 参考サイト1：OpenCVで丸い物体を検出
    # https://qiita.com/neriai/items/448a36992e308f4cabe2
    # 参考サイト2：HSVの色範囲を指定して画像にマスクする
    # http://rikoubou.hatenablog.com/entry/2019/02/21/190310

    hsvLower = np.array([mask_hsv[0], mask_hsv[2], mask_hsv[4]])
    hsvUpper = np.array([mask_hsv[1], mask_hsv[3], mask_hsv[5] ])

    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    hsvMask = cv2.inRange(hsv, hsvLower, hsvUpper)

    if debug_mode:
        cv2.imshow(name + 'mask', hsvMask)

    srcMasked = cv2.bitwise_and(src, src, mask=hsvMask)
    if debug_mode:
        cv2.imshow(name + 'srcMasked', srcMasked)
    # グレースケール画像へ変換
    gray = cv2.cvtColor(srcMasked, cv2.COLOR_BGR2GRAY)
    if debug_mode:
        cv2.imshow(name + 'gray', gray)

    # 2値化
    retval, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 輪郭を抽出
    #   contours : [領域][Point No][0][x=0, y=1]
    #   cv2.CHAIN_APPROX_NONE: 中間点も保持する
    #   cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 矩形検出された数（デフォルトで0を指定）
    detect_count = 0

    # 各輪郭の面積
    area = list(range(0, len(contours)))

    # 各輪郭に対する処理
    for i in range(0, len(contours)):

        # 輪郭の領域を計算
        area[i] = cv2.contourArea(contours[i])  # area[i]はi番目の面積、一次元リスト

        # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
        if area[i] < min_detect_area:  # or 1e8 < area[i]:
            continue

        # 外接矩形
        if len(contours[i]) > 0:
            rect = contours[i]
            x, y, w, h = cv2.boundingRect(rect)
            cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 外接矩形毎に画像を保存
            # cv2.imwrite('photos/' + str(detect_count) + '.jpg', src[y:y + h, x:x + w])
            detect_count = detect_count + 1

    # 一番大きい面積の風船を検出
    if len(area):  # 風船があるとき
        max_area_index = area.index(max(area))  # area
        if area[max_area_index] > min_detect_area:  # ゴミ（面積の小さい領域）をはじく。最も面積の大きいエリアの面積がしきい値以下なら風船はないと判断
            x, y, w, h = cv2.boundingRect(contours[max_area_index])
            ret = [int(x + w / 2), int(y + h / 2), area[max_area_index]]  # 風船の中心位置、面積を返す。
            # print("x:" + str(target_x) +", y:" + str(target_y))
            # 一番大きい風船の中心にマーカを描画
            cv2.drawMarker(src, (ret[0], ret[1]), color=(0, 255, 0), markerType=0, thickness=2)
        else:
            ret = None
    else:
        ret = None

    # 外接矩形された画像を表示
    cv2.imshow(name + 'output', src)
    # cv2.waitKey(0)

    return ret  # 風船がある時のみターゲットまでの座標(ピクセル数)を返す


# モータの制御量を決める関数
def controll_trace(_target):
    a = 0.0001 * abs(_target) # ルートをつけることで遠いところでの出力を抑える。
    if _target < 0:
        a = a *(-1)

    print(str(_target) + "a = " + str(a))
    motor_fwdcurve(20, a, 0)  # ここ


# カーブしながら直進する。valanceはパーセント＋で右、ーで左に進む。
# motor_fwdcurve(進行スピード、左右のハンドル量、潜水用スクリュー出力)
def motor_fwdcurve(forward_val, valance, down_val):
    motor_array = [0, forward_val + forward_val * valance, forward_val - forward_val * valance, 0, down_val]
    send_controlldata(motor_array)


# 風船上での左右移動する
def controll_on_balloon(_target_x, _target_y, _down):
    print("_target_x=" + str(_target_x) + ", _target_y:" + str(_target_y) + ", depth:" +
          str(math.pow((math.pow(_target_x, 2) + math.pow(_target_y, 2)), 2)))
    motor_horizontal_move(_target_x * 0.1, _target_y * 0.1,
                          0.3*(100 - 0.1 * math.pow(0.0001 * math.pow(_target_x, 2) + 0.0001 * math.pow(_target_y, 2),
                                               2)))  # ここ
    #motor_horizontal_move(_target_x * 0.3, _target_y * 0.3,20)  # ここ


# モータを左右方向に動かす
def motor_horizontal_move(x_val, y_val, down_val):
    motor_array = [0, 0, 0, 0, down_val]
    if x_val > 0:  # 右に行きたいとき。
        motor_array[2] = x_val
        motor_array[3] = x_val
    else:  # 左に行きたいとき
        motor_array[0] = abs(x_val)
        motor_array[1] = abs(x_val)

    if y_val < 0:  # 下に行きたいとき
        motor_array[1] = motor_array[1] + abs(y_val)
        motor_array[2] = motor_array[2] + abs(y_val)
    else:  # 上に行きたいとき
        motor_array[0] = motor_array[0] + abs(y_val)
        motor_array[3] = motor_array[3] + abs(y_val)

    send_controlldata(motor_array)


# raspiにモータの状態を送る関数
def send_controlldata(_send_data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        num = _send_data
        # サーバを指定
        s.connect((controll_raspi_ipaddr, controll_raspi_port))
        
        send_data = struct.pack('!B', int(0))
        for i in range(5):
            # モータ値が範囲内に収まるようにする
            if num[i] < 0:
                num[i] = 0
            elif num[i] > max_motor_val:
                num[i] = max_motor_val

            bin = struct.pack('!B', int(num[i] * motor_power_const)) 
            send_data = send_data + bin

        if debug_mode:
            print(num)
        s.sendall(send_data)



# ここからメインループ

mode = 0
cnt2 = 0

while True:

    # カメラから映像を取得する
    img = getimage(camera_raspi_ipaddr, camera1_raspi_port)
    if debug_mode:
        cv2.imshow('getimage',img)  # ラズパイから送られてきた生画像を表示
    target = detect_contour(img, mask_hsv_cam1, 'cam1:')

    img2 = getimage(camera_raspi_ipaddr, camera2_raspi_port)
    if debug_mode:
        cv2.imshow('cam2:getimage', img2)
    target2 = detect_contour(img2,mask_hsv_cam2,  'cam2:')

    if mode == 0:  # 風船を見つけに行くモード
        if target != None:  # 風船を検出している時
            controll_trace(target[0] - camera1_resolution[0] / 2)   # 風船をトレースする。target[0] - camera1_resolution[0] / 2は中心からの位置
            cnt2 = 0
        else:  # 風船が見つからないとき
            print("balloon is not detected")
            cnt2 = cnt2 + 1

            # 風船探索モード
            if cnt2 < 30:       # 直進
                stop_array = [0, 10, 10, 0, 0] 　# モータの出力を0-100で指定する。左前、左後、右後、右前、上下の順で指定する。
                send_controlldata(stop_array)
            elif cnt2 < 90:     # 左回転
                stop_array = [2, 0, 2, 0, 0]
                send_controlldata(stop_array)
            elif cnt2 < 170:    # 右回転
                stop_array = [0, 2, 0, 2, 0]
                send_controlldata(stop_array)
            else:
                cnt2 = 0

        if target2 != None:  # 下カメラで風船を検出したとき
            if target2[2] > camera2_balloon_detection_threshold:    # 面積がしきい値より大きいときに、風船割るモードに遷移
                cnt = 0
                #mode = 1
                print("下の風船を割るモードへ")

    elif mode == 1:  # 風船を割るモード
        if target2 != None:  # 風船があるとき
            controll_on_balloon(target2[0] - camera2_resolution[0] / 2, target2[1] - camera2_resolution[1] / 2, 0)  # 風船をトラックする。
            cnt = 0
        else:  # 風船を見失ったとき（風船がないとき）
            print("under balloon is not detected")
            cnt = cnt + 1

            stop_array = [0, 0, 0, 0, 0]
            send_controlldata(stop_array)

        if cnt > Balloon_non_detection_waiting_time:    # 長時間見つけられないとき、風船が割れたときは風船を探すモードに戻る
            mode = 0
            print("通常モードへ")

   # sを入力すればカメラの画像を保存する
    wait_key = cv2.waitKey(1)
    if wait_key & 0xFF == ord('s'):
        time = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        success_check1 = cv2.imwrite(time + '_image1.jpg', img)
        success_check2 = cv2.imwrite(time + '_image2.jpg', img2)
        if success_check1 == False:
            print("failed")
        if success_check2 == False:
            print("failed")

    # qを入力すれば処理終了
    elif wait_key & 0xFF == ord('q'):
        stop_array = [0, 0, 0, 0, 0]    # モータを止める
        send_controlldata(stop_array)
        break