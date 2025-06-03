import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import argparse
import matplotlib.patches as mpatches
import glob
import cv2
import os
from tqdm import tqdm

def plot_color_bar(label, class_names):
    label = label.astype(np.int32)
    # ラベルを2D画像に整形（ここでは縦5pxのカラーバーを作る）
    bar_height = 5
    label_array = np.tile(label, (bar_height, 1))  # (5, 時間ステップ数)

    # カラーマップを作成（クラス数に合わせて変更）
    num_classes = len(class_names)
    base_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'yellow', 'magenta', 'brown', 'gray']
    cmap = ListedColormap(base_colors[:num_classes])

    # 描画
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.imshow(label_array, aspect='auto', cmap=cmap)
    ax.axis('off')

    handles = [mpatches.Patch(color=base_colors[i], label=class_names[i]) for i in range(len(class_names))]
    ax.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.5),
        ncol=len(class_names),
        frameon=False,
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig("{}/{}/color_map.png".format(args.out, args.video_name))

def make_video_with_class(labels, class_names, output_path, fps=60):
    frame_list = glob.glob("/home/skato/work/WSC/data/Avi_data/Crop/{}/*.png".format(args.video_name))
    
    first_image = cv2.imread(frame_list[0])
    height, width, _ = first_image.shape
        
    # 動画ライターを設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' や 'avc1' も可
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height+100))

    for filename, clas in tqdm(zip(labels[:,0], labels[:,1]), desc="画像から動画を作成中"):
        img_path = [name for name in frame_list if filename in name]
        img = cv2.imread(img_path[0])
        
        img_ = caption_img(img, clas, class_names)

        if img is None:
            print(f"スキップ: {img_path}")
            continue
        out.write(img_)

    out.release()
    print(f"✅ 動画を保存しました: {output_path}")


def caption_img(img, clas, class_names):
    num_classes = len(class_names)    
    colors = [
        (0, 0, 255),      # red
        (0, 255, 0),      # green
        (255, 0, 0),      # blue
        (0, 165, 255),    # orange
        (128, 0, 128),    # purple
        (255, 255, 0),    # cyan (OpenCV BGRならこれが黄色)
        (0, 255, 255),    # yellow (BGRで黄)
        (255, 0, 255),    # magenta
        (42, 42, 165),    # brown (approximate)
        (128, 128, 128)   # gray
    ]

    height, width = img.shape[:2]
    bar_height = 100

    # グレー色 (BGR)
    gray = np.array([200, 200, 200], dtype=np.uint8)
    
    # カラーバー画像を初期化
    color_bar = np.zeros((bar_height, width, 3), dtype=np.uint8)

    # 1クラスあたりの幅
    class_width = width // num_classes

    for i in range(num_classes):
        start_x = i * class_width
        # 最後の区画は画像の端まで塗る（割り切れない幅に対応）
        end_x = (i + 1) * class_width if i < num_classes - 1 else width
        if i == int(clas):
            color_bar[:, start_x:end_x] = colors[i]
        else:
            color_bar[:, start_x:end_x] = gray
        
        # クラス名を描画
        text = class_names[i]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.4
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = start_x + (class_width - text_w) // 2
        text_y = (bar_height + text_h) // 2  # 垂直中央に来るよう調整

        text_color = (255, 255, 255) if i == clas else (0, 0, 0)
        cv2.putText(color_bar, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)


    # 画像の下にカラーバーを縦連結
    img_with_bar = np.vstack([img, color_bar])
    #cv2.imwrite("xxxxx.png", img_with_bar)
    return img_with_bar

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMLoss')
    parser.add_argument('--out', type=str, default='result_la')
    parser.add_argument('--video_name', type=str, default='juntendo-room12-20210517-122015')
    args = parser.parse_args()

    class_names = ["washing", "indigocarmine", "bleeding", "other", "normal"]

    # 例: 各フレームのアクションラベル（0～4 のクラス）
    labels = np.loadtxt("{}/{}/prediction.txt".format(args.out, args.video_name), dtype=str)
    label = labels[:, 1]
    
    # plot color bar
    plot_color_bar(label, class_names)

    # make video
    make_video_with_class(labels, class_names, "{}/{}/video.mp4".format(args.out, args.video_name))
