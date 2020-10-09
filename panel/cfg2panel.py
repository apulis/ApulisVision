import mmcv
import json
from mmcv import Config


def recureConfig(config):
    nodes = []
    for key, value in config.items():
        if key == 'model':
            if isinstance(value, dict):
                for key1, value1 in value.items():
                    if isinstance(value1, dict):
                        node1 = {'name': key1, 'config': [], 'children': []}
                        for key2, value2 in value1.items():
                            if isinstance(value2, dict):
                                node2 = {
                                    'name': key2,
                                    'config': [],
                                    'children': []
                                }
                                for key3, value3 in value2.items():
                                    if isinstance(value3, dict):
                                        node3 = {
                                            'name': key3,
                                            'config': [],
                                            'children': []
                                        }
                                        for key4, value4 in value3.items():
                                            if isinstance(value4, dict):
                                                node3 = {
                                                    'name': key4,
                                                    'config': [],
                                                    'children': []
                                                }
                                                for key5, value5 in value4.items(
                                                ):
                                                    pass
                                            else:
                                                if type(value4) in ['int', 'float']:
                                                    node3['config'].append({
                                                        'key': key4,
                                                        'value': value4,
                                                        'type': 'number'
                                                    })
                                                else:
                                                    node3['config'].append({
                                                        'key': key4,
                                                        'value': value4,
                                                        'type':
                                                        'string'
                                                    })                                            
                                        node2['children'].append(node3)
                                    else:
                                        if type(value3).__name__ in ['int', 'float']:
                                            node2['config'].append({
                                                'key': key3,
                                                'value': value3,
                                                'type': 'number'
                                            })
                                        else:
                                            node2['config'].append({
                                                'key': key3,
                                                'value': value3,
                                                'type': 'string'
                                            })
                                node1['children'].append(node2)
                            else:
                                if type(value2).__name__ in ['int', 'float']:
                                    node1['config'].append({
                                        'key': key2,
                                        'value': value2,
                                        'type': 'number'
                                    })
                                else:
                                    node1['config'].append({
                                        'key': key2,
                                        'value': value2,
                                        'type': 'string'
                                    })
                        nodes.append(node1)
    return nodes


                                  
if __name__ == '__main__':
    cfg_file = '../configs_mmcls/_base_/models/resnet50.py'
    cfg_file = '../configs_mmcls/_base_/models/mobilenet_v2_1x.py'
    cfg_file = '../configs_mmcls/_base_/models/resnet50_cifar.py'
    cfg_file = '../configs_custom/mmdet/faster_rcnn_r50_fpn_1x_coco.py'
    cfg_file = '../configs_mmseg/_base_/models/fcn_r50-d8.py'
    cfg = Config._file2dict(cfg_file)
    nodes = recureConfig(cfg[0])
    with open("seg.json", "w") as out_file:
        json.dump(nodes, out_file, indent=4)
    out_file.close()