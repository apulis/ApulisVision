import json
import MySQLdb
import mmcv
from mmcv import Config


def genNodes(panelJson):
    nodes = []
    for idx, item in enumerate(panelJson):
        node = {}
        if item != {}:
            node["id"] = item['name']
            node["idx"] = idx
            # 获取第一个乐高块的name
            node["name"] = [i for i in item["children"][0].items()][0][0]
            # 获取第一个乐高块的configs
            configs = [i for i in item["children"][0].items()][0][1]
            node["config"] = []
            for config in configs:
                node["config"].append({
                    "key": config["key"],
                    "type": config["type"],
                    "value": config["value"],
                })
        else:
            node = {
                "id": "Input",
                "name": "dog-vs-cat",
                "config": [
                    {
                        "key": "data_path",
                        "type": "disabled",
                        "value": "/data/dataset/storage/dog-vs-cat/"
                    }
                ],
                "idx": 0,
            }
        nodes.append(node)
    return nodes


def genEdges(nodes):
    edges = []
    for idx, item in enumerate(nodes):
        if idx == len(nodes) - 1:
            break
        edges.append({
            "source": f'{item["id"]}-{item["name"]}',
            "target": f'{nodes[idx + 1]["id"]}-{nodes[idx + 1]["name"]}'
        })
    return edges


def insertSql(modelName, fileName, modelUse, params):
    db = MySQLdb.connect("localhost", "root", "root", "ai_arts", charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    deleteSql = '''delete
    from modelsets
    where is_advance = 1 and `use` like 'Avisualis%';'''

    deleteSql = f'''delete from modelsets where name = '{modelName}';'''
    cursor.execute(deleteSql)

    insertSql = f'''INSERT INTO `modelsets` (`created_at`, `updated_at`, `deleted_at`, `name`, `description`, `creator`, `version`,
                         `status`, `size`, `use`, `job_id`, `data_format`, `dataset_name`, `dataset_path`, `params`,
                         `engine`, `precision`, `is_advance`, `code_path`, `param_path`, `output_path`, `startup_file`,
                         `evaluation_id`, `device_type`, `device_num`)
VALUES ('2020-09-20 13:40:13', '2020-09-20 03:40:13', NULL, '{modelName}',
        '',
        'avisualis', '0.0.1', 'normal', 0,
        'Avisualis_{modelUse}', '', '', '', '','{params}', 'apulistech/apulisvision', '', 1, '/data/premodel/code/ApulisVision/', '/home/admin/avisualis/',
        '/home/admin/avisualis/{modelUse}', '/data/premodel/code/ApulisVision/tools/train_{fileName}.py', '', 'nvidia_gpu_amd64', 1)
        '''

    try:
        # 执行sql语句
        cursor.execute(insertSql)
        # 提交到数据库执行
        db.commit()
    except Exception as e:
        print(e)
        # 发生错误时回滚
        db.rollback()
    db.close()


nodes = []
configs = []

import json


def recureConfig(config):
    for key, value in config.items():
        if key == 'model':
            if isinstance(value, dict):
                for key1, value1 in value.items():
                    if isinstance(value1, dict):
                        node1 = {
                            "name": key1,
                            "config": [],
                            "children": []
                        }
                        for key2, value2 in value1.items():
                            if isinstance(value2, dict):
                                node2 = {
                                    "name": key2,
                                    "config": [],
                                    "children": []
                                }
                                for key3, value3 in value2.items():
                                    if isinstance(value3, dict):
                                        node3 = {
                                            "name": key3,
                                            "config": [],
                                            "children": []
                                        }
                                        for key4, value4 in value3.items():
                                            if isinstance(value4, dict):
                                                node3 = {
                                                    "name": key4,
                                                    "config": [],
                                                    "children": []
                                                }
                                                for key5, value5 in value4.items():
                                                    pass
                                            else:
                                                node3['config'].append({
                                                    "key": key4,
                                                    "value": value4,
                                                    "type": "string"
                                                })
                                        node2['children'].append(node3)

                                    else:
                                        node2['config'].append({
                                            "key": key3,
                                            "value": value3,
                                            "type": "string"
                                        })
                                node1['children'].append(node2)

                            else:
                                node1['config'].append({
                                    "key": key2,
                                    "value": value2,
                                    "type": "string"
                                })
                        nodes.append(node1)


if __name__ == '__main__':
    modelName = "More_Classfication"
    fileName = "cls"
    modelUse = "Model_Classfication"
    cfg = Config._file2dict("t.py")
    recureConfig(cfg[0])

    print(json.dumps(nodes, indent=1))

    # with open("faster_rcnn_r50_fpn_1x_coco.py")as f1:
    #     panelJson = json.load(f1)
    #     nodes = genNodes(panelJson)
    #     edges = genEdges(nodes)
    #     nodes = json.dumps(nodes, sort_keys=True)
    #     edges = json.dumps(edges, sort_keys=True)
    #     panel = json.dumps(panelJson, sort_keys=True)
    #     params = json.dumps({
    #         "panel": panel,
    #         "nodes": nodes,
    #         "edges": edges
    #     })
    #     params = params.replace('\\"', '\\\\"').replace(' ', '')
    #     print(params)
    # insertSql(modelName, fileName, modelUse, params)
