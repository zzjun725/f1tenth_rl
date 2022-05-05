import cv2
import fire
import yaml
import os

work_dir = os.path.abspath('..')
cfg_dir = os.path.abspath('.')
map_dir = os.path.join(cfg_dir, 'maps')
sim_cfg_temp_file = './maps/config_example_map.yaml'


def pgm2png(rawMap_name='levine_test4'):
    rawMap_cfg = os.path.join(map_dir, rawMap_name + '.yaml')
    rawMap_pic_file = os.path.join(map_dir, rawMap_name + '.pgm')
    rawMap_png = cv2.imread(rawMap_pic_file, cv2.IMREAD_GRAYSCALE)

    cv2.imwrite(os.path.join(map_dir, rawMap_name+'.png'), rawMap_png)

    with open(rawMap_cfg) as f:
        raw_map_cfg = yaml.safe_load(f)

    with open(sim_cfg_temp_file) as f:
        sim_cfg_temp = yaml.safe_load(f)

    sim_cfg_temp['map_path'] = os.path.join(map_dir, rawMap_name)
    sim_cfg_temp['wpt_path'] = os.path.join(map_dir, rawMap_name+'_wp.csv')

    with open(os.path.join(map_dir, 'config_' + rawMap_name + '.yaml'), 'w') as f:
        yaml.dump(sim_cfg_temp, f)


if __name__ == '__main__':
    fire.Fire({
        'get_pngmap': pgm2png
    })




