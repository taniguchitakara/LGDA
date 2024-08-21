import json

def main(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    id_file_map = {image['id']: image['file_name'] for image in data['images']}
    
    with open(output_file, 'w') as f:
        json.dump(id_file_map, f, indent=4)

if __name__ == "__main__":
    input_file = "data/manga/voc_annotation/lovehina_vol14.json"  # 入力JSONファイル名
    output_file = "output_id.json"  # 出力JSONファイル名
    main(input_file, output_file)
