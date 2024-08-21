import json
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

def draw_bboxes(image_path, bboxes, output_path, font_size=15, bbox_thickness=2, score_threshold=0.5):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/large/ttani_2/bhrl/lovehina_be_visualized/arial.ttf", font_size)

    for bbox in bboxes:
        x, y, w, h = bbox['bbox']
        category_id = bbox['category_id']
        score = bbox['score']

        if score >= score_threshold:
            # Assign different colors based on category_id
            color = get_category_color(category_id)

            draw.rectangle([x, y, x + w, y + h], outline=color, width=bbox_thickness)
            draw.text((x, y), f"Label: {category_id} Score: {score:.2f}", font=font, fill=color)

    image.save(output_path)

def get_category_color(category_id):
    # Define colors based on category_id
    color_mapping = {
    1: 'red',
    2: 'green',
    3: 'blue',
    4: 'orange',
    5: 'purple',
    6: 'yellow',
    7: 'cyan',
    8: 'pink',
    9: 'brown',
    10: 'grey',
    11: 'lightblue',
    12: 'lime',
    13: 'maroon',
    14: 'navy',
    15: 'olive',
    16: 'indigo',
    17: 'teal',
    18: 'violet',
    19: 'salmon',
    20: 'tan',
    21: 'khaki',
    22: 'orchid',
    23: 'peru',
    24: 'plum',
    25: 'gold',
    26: 'silver',
    27: 'crimson',
    28: 'tomato',
    29: 'magenta',
    30: 'thistle',
    31: 'azure',
    32: 'darkgreen',
    33: 'skyblue',
    34: 'limegreen',
    35: 'darkred',
    36: 'darkblue',
    37: 'hotpink',
    38: 'darkcyan',
    39: 'darkviolet',
    40: 'lightgreen',
    41: 'darkorange',
    42: 'coral',
    43: 'seagreen',
    44: 'mediumblue',
    45: 'springgreen',
    46: 'slategray',
    47: 'orchid',
    48: 'orangered',
    49: 'deeppink',
    50: 'lime',
    51: 'sandybrown',
    52: 'cadetblue',
    53: 'midnightblue',
    54: 'lightcoral',
    55: 'steelblue',
    56: 'rosybrown',
    57: 'mediumseagreen',
    58: 'darkslategray',
    59: 'dodgerblue',
    60: 'darkgoldenrod',
    61: 'firebrick',
    62: 'chocolate',
    63: 'indianred',
    64: 'mediumorchid',
    65: 'lightseagreen',
    66: 'mediumvioletred',
    67: 'olivedrab',
    68: 'darkorchid',
    69: 'goldenrod',
    70: 'lightskyblue',
    71: 'palegreen',
    72: 'royalblue',
    73: 'darkslateblue',
    74: 'greenyellow',
    75: 'mediumturquoise',
    76: 'cornflowerblue',
    77: 'darkkhaki',
    78: 'powderblue',
    79: 'sienna',
    80: 'mediumslateblue',
    81: 'darkturquoise',
    82: 'lightgray',
    83: 'burlywood',
    84: 'darkgray',
    85: 'mediumaquamarine',
    86: 'darkolivegreen',
    87: 'saddlebrown',
    88: 'darkmagenta',
    89: 'darkseagreen',
    90: 'darkslategray',
    91: 'mediumspringgreen',
    92: 'lightsteelblue',
    93: 'darkred',
    94: 'navajowhite',
    95: 'darkslateblue',
    96: 'mediumblue',
    97: 'darkslateblue',
    98: 'slateblue',
    99: 'peru',
    100: 'darkred',
    # Add more colors as needed
    }
    # Default to white if category_id not in color_mapping
    return color_mapping.get(category_id, 'white')

def process_images(json_file_path, image_folder, output_folder,id_image_cor_path, font_size=15, bbox_thickness=2, score_threshold=0.5):
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    with open(id_image_cor_path, 'r') as f:
        id_image_cor = json.load(f)

    image_ids = set()
    for bbox_data in json_data:
        image_ids.add(bbox_data['image_id'])

    os.makedirs(output_folder, exist_ok=True)

    for image_id in tqdm(image_ids, desc='Processing Images'):

        image_path = os.path.join(image_folder,id_image_cor[str(image_id)])
        output_path = os.path.join(output_folder, f"Output_{str(image_id).zfill(3)}.jpg")

        bboxes = [bbox for bbox in json_data if bbox['image_id'] == image_id]

        draw_bboxes(image_path, bboxes, output_path, font_size=font_size, bbox_thickness=bbox_thickness, score_threshold=score_threshold)

if __name__ == "__main__":
    json_file_path = 'lovehina_results_unseen.pkl.bbox.json'
    id_image_cor_path = "output_id.json"
    image_folder = './'
    output_folder = "./unseen_folder"
    font_size = 64
    bbox_thickness = 5
    score_threshold = 0.50

    process_images(json_file_path, image_folder, output_folder,id_image_cor_path, font_size=font_size, bbox_thickness=bbox_thickness, score_threshold=score_threshold)

