import json

import MySQLdb


def genNodes(panelJson):
    nodes = []
    for idx, item in enumerate(panelJson):
        node = {}
        if item != {}:
            node['id'] = item['name']
            node['idx'] = idx
            # 获取第一个乐高块的name
            node['name'] = [i for i in item['children'][0].items()][0][0]
            # 获取第一个乐高块的configs
            configs = [i for i in item['children'][0].items()][0][1]
            node['config'] = []

            for config in configs:
                node['config'].append({
                    'key': config['key'],
                    'type': config['type'],
                    'value': config['value'],
                })
        else:
            node = {
                'id':
                'Input',
                'name':
                'dog-vs-cat',
                'config': [{
                    'key': 'data_path',
                    'type': 'disabled',
                    'value': '/data/dataset/storage/dog-vs-cat/'
                }],
                'idx':
                0,
            }
        nodes.append(node)
    print(nodes)
    return nodes


def genEdges(nodes):
    edges = []
    for idx, item in enumerate(nodes):
        if idx == len(nodes) - 1:
            break
        edges.append({
            'source':
            f'{item["id"]}-{item["name"]}',
            'target':
            f'{nodes[idx + 1]["id"]}-{nodes[idx + 1]["name"]}'
        })
    return edges


def insertSql(modelName, fileName, modelUse, params):
    db = MySQLdb.connect("localhost", "root", "root", "ai_arts", charset='utf8')
    # db = MySQLdb.connect(host="219.133.167.42", port=53306, user="root", password="apulis#2019#wednesday", db="ai_arts",
    #                      charset='utf8')

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


if __name__ == '__main__':
    modelName = "More_recur_Classfication"
    fileName = "cls"
    modelUse = "Model_recur_Classfication"
    with open("det_panel.json")as f1:
        panelJson = json.load(f1)
        # nodes = genNodes(panelJson)
        # edges = genEdges(nodes)
        # nodes = json.dumps(nodes, sort_keys=True)
        # edges = json.dumps(edges, sort_keys=True)
        panel = json.dumps(panelJson, sort_keys=True)
        params = json.dumps({
            'panel': panel,
            # "nodes": nodes,
            # "edges": edges
        })
        params = params.replace('\\"', '\\\\"').replace(' ', '')
        print(params)
    insertSql(modelName, fileName, modelUse, params)
