import argparse
import random
import sys
from collections import defaultdict

from tqdm import tqdm

sys.path.append('./code')

from code.executor import *
from code.interpreter import *
from code.methods import *
from code.segment_anything import sam_model_registry

METHODS_MAP = {
    "baseline": Baseline,
    "random": Random,
    "parse": Parse,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='./data/val/annos.jsonl',
                        help="input file with expressions and annotations in jsonlines format")
    parser.add_argument("--image_root", type=str, default='./data/val/images', help="path to images")
    parser.add_argument("--clip_model", type=str, default="RN50x16,ViT-B/32",
                        help="which clip model to use (should use RN50x16, ViT-B/32, or both separated by a comma")
    # parser.add_argument("--albef_path", type=str, default=None,
    #                     help="to use ALBEF (instead of CLIP), specify the path to the ALBEF checkpoint")
    parser.add_argument("--method", type=str, default="parse", help="method to solve expressions")
    # 使用shade反而降低
    parser.add_argument("--box_representation_method", type=str, default="crop,blur",
                        help="method of representing boxes as individual images (crop, blur, or both separated by a comma)")

    # 是否使用SAM
    parser.add_argument("--use_sam", type=bool, default=True, help="如果使用SAM，则设置为True")
    # 是否考虑颜色
    parser.add_argument("--use_color", type=bool, default=True, help="是否加入颜色计算权重")
    # 设置SAM生成pt的路径
    parser.add_argument("--sam_pt_path", type=str,
                        default=r"code/sam_pt",
                        help="如果使用sam并且已有对image生成的pt文件，可以直接设置路径。设置路径后将直接读取pt文件，而不再调用sam模型")

    parser.add_argument("--sam_checkpoint", type=str,
                        default=r"E:\全球校园人工智能算法精英大赛\算法挑战赛文件\SAM\SAM模型权重\sam_vit_h_4b8939.pth")
    parser.add_argument("--box_method_aggregator", type=str, default="sum",
                        help="method of combining box representation scores")
    parser.add_argument("--box_area_threshold", type=float, default=0.045,
                        help="minimum area (as a proportion of image area) for a box to be considered as the answer")
    parser.add_argument("--results_path", type=str, default='./code/result/result.json',
                        help="(optional) output path to save results")
    parser.add_argument("--detector_file", type=str, default='./data/dets_dict.json',
                        help="(optional) file containing object detections. if not provided, the gold object boxes will be used.")
    parser.add_argument("--mock", action="store_true", help="(optional) mock CLIP execution.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device to use.")
    parser.add_argument("--shuffle_words", action="store_true", help="If true, shuffle words in the sentence")
    parser.add_argument("--gradcam_alpha", type=float, nargs='+', help="alpha value to use for gradcam method")
    parser.add_argument("--enlarge_boxes", type=float, default=0.2,
                        help="(optional) whether to enlarge boxes when passing them to the model")

    parser.add_argument("--part", type=str,
                        # default='100,38',
                        help="(optional) specify how many parts to divide the dataset into and which part to run in the format NUM_PARTS,PART_NUM")

    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of instances to process in one model call (only supported for baseline model)")
    parser.add_argument("--baseline_head", action="store_true",
                        help="For baseline, controls whether model is called on both full expression and head noun chunk of expression")
    parser.add_argument("--mdetr", type=str, default=None,
                        help="to use MDETR as the executor model, specify the name of the MDETR model")
    parser.add_argument("--albef_block_num", type=int, default=8, help="block num for ALBEF gradcam")
    parser.add_argument("--albef_mode", type=str, choices=["itm", "itc"], default="itm")
    parser.add_argument("--expand_position_embedding", action="store_true", default=False)
    parser.add_argument("--gradcam_background", action="store_true")
    parser.add_argument("--mdetr_given_bboxes", action="store_true")
    parser.add_argument("--mdetr_use_token_mapping", action="store_true")
    parser.add_argument("--non_square_size", action="store_true")
    parser.add_argument("--blur_std_dev", type=int, default=100, help="standard deviation of Gaussian blur")
    parser.add_argument("--gradcam_ensemble_before", action="store_true",
                        help="Average gradcam maps of different models before summing over the maps")
    parser.add_argument("--cache_path", type=str, default='./code/result/sam_blur_and_crop_threshold(0)',
                        help="cache features")
    # Arguments related to Parse method.
    parser.add_argument("--no_rel", action="store_true", help="Disable relation extraction.")
    parser.add_argument("--no_sup", action="store_true", help="Disable superlative extraction.")
    parser.add_argument("--no_null", action="store_true", help="Disable null keyword heuristics.")
    parser.add_argument("--ternary", action="store_true", help="Disable ternary relation extraction.")

    # 2 10.2 67
    parser.add_argument("--baseline_threshold", type=float, default=2,
                        help="(Parse) Threshold to use relations/superlatives.")
    parser.add_argument("--temperature", type=float, default=5, help="(Parse) Sigmoid temperature.")
    parser.add_argument("--superlative_head_only", action="store_true",
                        help="(Parse) Superlatives only quanntify head predicate.")
    # false使用的是softmax
    parser.add_argument("--sigmoid", action="store_true", help="(Parse) Use sigmoid, not softmax."
                        , default=True)

    parser.add_argument("--no_possessive", action="store_true",  # default=True,
                        help="(Parse) Model extraneous relations as possessive relations.")
    parser.add_argument("--expand_chunks", action="store_true",  # default=False,
                        help="(Parse) Expand noun chunks to include descendant tokens that aren't ancestors of tokens in other chunks")
    parser.add_argument("--parse_no_branch", action="store_true",  # default=False,
                        help="(Parse) Only do the parsing procedure if some relation/superlative keyword is in the expression")
    parser.add_argument("--possessive_no_expand", action="store_true", help="(Parse) Expand ent2 in possessive case")
    args = parser.parse_args()

    with open(args.input_file) as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    executor = ClipExecutor(clip_model=args.clip_model, box_representation_method=args.box_representation_method,
                            method_aggregator=args.box_method_aggregator, device=device,
                            square_size=not args.non_square_size,
                            expand_position_embedding=args.expand_position_embedding, blur_std_dev=args.blur_std_dev,
                            cache_path=args.cache_path)

    print(args.method)
    method = METHODS_MAP[args.method](args)
    correct_count = 0
    total_count = 0
    if args.detector_file:
        detector_file = open(args.detector_file)
        detections_list = json.load(detector_file)
        if isinstance(detections_list, dict):
            detections_map = {int(image_id): detections_list[image_id] for image_id in detections_list}
        else:
            detections_map = defaultdict(list)
            for detection in detections_list:
                detections_map[detection["image_id"]].append(detection["box"])
    if args.part is not None:
        num_parts = int(args.part.split(",")[0])
        part = int(args.part.split(",")[1])
        data = data[int(len(data) * part / num_parts):int(len(data) * (part + 1) / num_parts)]
    batch_pred_box = []

    all_direct_list = ["left", "west", "right", "east", "above", "north", "top", "back", "behind", "below", "south",
                       "under", "front"]

    # 如果使用sam，则检查是否已经存在路径
    sam_pt_path = args.sam_pt_path
    sam = None
    if args.use_sam:
        if sam_pt_path is None:  # 检查路径是否存在
            print("使用了SAM但是没有正确的设置.pt文件路径，请重新设置参数【--sam_pt】")
            sys.exit()
        else:
            sam_pt_files = os.listdir(sam_pt_path)
            # 配置sam
            model_type = "vit_h"
            sam_checkpoint = args.sam_checkpoint
            device = "cuda"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

    count = 0
    for datum in tqdm(data):
        file_name = datum["file_name"]
        img_path = os.path.join(args.image_root, file_name)
        img = Image.open(img_path).convert('RGB')
        image_id = datum["image_id"]

        # # todo 单个题目DEBUG
        # if image_id != 27237:
        #     continue

        sentence = ''
        for simple_sentence in datum["sentences"]:
            # simple_sentence = simple_sentence.replace("right", "east")
            # simple_sentence = simple_sentence.replace("left",  "west")
            sentence += simple_sentence + '\n'

        # for sentence in datum["sentences"]:
        # sentence = "at 3:00 pm\n"

        boxes = [Box(x=box[0], y=box[1], w=box[2], h=box[3]) for box in detections_map[int(datum["image_id"])]]
        if len(boxes) == 0:
            boxes = [Box(x=0, y=0, w=img.width, h=img.height)]

        # todo 读取sam_mask
        sam_masks = None
        if args.use_sam:
            sam_file_name = f"{str(image_id)}.pt"
            sam_pt_data = None
            if sam_file_name in sam_pt_files:  # 如果有此图片的sam的pt文件缓存，则加载
                sam_pt_data = torch.load(args.sam_pt_path + "/" + sam_file_name)
            sam_masks = None if sam_pt_data is None else sam_pt_data['masks']

        env = Environment(img, boxes, executor, (args.mdetr is not None and not args.mdetr_given_bboxes),
                          str(datum["image_id"]), sam_masks=sam_masks, use_color=args.use_color,
                          image_path=args.image_root, dets_dict_path=args.detector_file,
                          sam=sam)

        if args.shuffle_words:
            words = sentence.lower().split()
            random.shuffle(words)
            result = method.execute(" ".join(words), env)
        else:
            result = method.execute(sentence.lower(), env)
        boxes = env.boxes

        for _ in range(len(datum["sentences"])):
            pred_box = boxes[result["pred"]]
            batch_pred_box.append([pred_box.left, pred_box.top, pred_box.right, pred_box.bottom])

    with open(args.results_path, 'w') as f:
        json.dump(batch_pred_box, f)
